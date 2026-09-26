# -*- coding: utf-8 -*-
"""独立复核 Q4 的分区完整性、任务继承和组内峰值资源。"""

from __future__ import annotations

import csv
import math
import os
import sys
from collections import Counter

from core import charge_time_s, load_aircraft_types, load_fleet, load_relay_fleet, load_relay_type

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, "结果")
SITES = {f"S{i:03d}" for i in range(1, 16)}
RESOURCE_COLUMNS = (
    "A型运输无人机数", "B型运输无人机数", "C型运输无人机数",
    "A型电池组数", "B型电池组数", "C型电池组数",
    "中继无人机数", "中继能源组件数",
)


def read_csv(name: str) -> list[dict[str, str]]:
    with open(os.path.join(RESULT_DIR, name), encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def peak(intervals: list[tuple[float, float]]) -> int:
    events = []
    for start, end in intervals:
        events.extend(((round(start, 3), 1), (round(end, 3), -1)))
    active = answer = 0
    for _, change in sorted(events, key=lambda item: (item[0], item[1])):
        active += change
        answer = max(answer, active)
    return answer


def close(actual: float, expected: float, tolerance: float = 1e-6) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"数值不一致：{actual} != {expected}")


def main() -> None:
    aircraft = load_aircraft_types()
    drones, battery_stock, full_charge = load_fleet()
    relay_type = load_relay_type()
    relay_drones, relay_components, relay_full_charge = load_relay_fleet()
    drone_stock = {g: sum(kind == g for _, kind in drones) for g in "ABC"}
    inventory = (drone_stock["A"], drone_stock["B"], drone_stock["C"],
                 battery_stock["A"], battery_stock["B"], battery_stock["C"],
                 len(relay_drones), relay_components)

    trips = []
    for row in read_csv("Q3_运输架次.csv"):
        kind = row["机型编号"]
        start, end = float(row["开始时刻s"]), float(row["返回O01时刻s"])
        energy = float(row["架次能耗kWh"])
        trips.append({
            "id": row["架次编号"], "sites": set(row["访问服务区顺序"].split(",")),
            "kind": kind, "start": start, "end": end, "energy": energy,
            "battery_end": end + charge_time_s(full_charge[kind], 1.0 - energy / aircraft[kind].battery_kwh),
        })

    relays = {}
    for row in read_csv("Q3_中继架次.csv"):
        start, end = float(row["开始时刻s"]), float(row["返回O01时刻s"])
        energy = float(row["架次能耗kWh"])
        relays[row["中继架次编号"]] = {
            "start": start, "end": end, "energy": energy,
            "drone_end": end + relay_type["turnaround_s"],
            "component_end": end + charge_time_s(
                relay_full_charge, 1.0 - energy / relay_type["battery_kwh"]),
        }

    trip_relays = {trip["id"]: set() for trip in trips}
    for row in read_csv("Q3_通信保障.csv"):
        if row["保障方式"] == "中继":
            trip_relays[row["运输架次编号"]].add(row["中继架次编号"])

    box_sites = Counter(row["服务区编号"] for row in read_csv("Q3_逐箱交付.csv"))
    config = read_csv("Q4_分区配置.csv")
    comparison = {int(row["K"]): row for row in read_csv("Q4_方案比较.csv")}
    if Counter(int(row["K"]) for row in config) != Counter({2: 2, 3: 3}):
        raise AssertionError("Q4 分组行数不正确")

    for k in (2, 3):
        rows = [row for row in config if int(row["K"]) == k]
        memberships = [set(filter(None, row["服务区列表"].split(","))) for row in rows]
        if any(not group for group in memberships):
            raise AssertionError(f"K={k} 存在空组")
        if set().union(*memberships) != SITES or sum(map(len, memberships)) != len(SITES):
            raise AssertionError(f"K={k} 的服务区不是互斥完备划分")

        allocation = [0] * len(RESOURCE_COLUMNS)
        workloads = []
        box_counts = []
        for row, sites in zip(rows, memberships):
            selected = [trip for trip in trips if trip["sites"] & sites]
            if any(not trip["sites"] <= sites for trip in selected):
                raise AssertionError(f"K={k} 拆分了多服务区运输架次")
            relay_ids = set().union(*(trip_relays[trip["id"]] for trip in selected)) if selected else set()
            drone_need = [peak([(t["start"], t["end"]) for t in selected if t["kind"] == g]) for g in "ABC"]
            battery_need = [peak([(t["start"], t["battery_end"]) for t in selected if t["kind"] == g]) for g in "ABC"]
            relay_drone_need = peak([(relays[r]["start"], relays[r]["drone_end"]) for r in relay_ids])
            component_need = peak([(relays[r]["start"], relays[r]["component_end"]) for r in relay_ids])
            expected = tuple(drone_need + battery_need + [relay_drone_need, component_need])
            actual = tuple(int(row[column]) for column in RESOURCE_COLUMNS)
            if actual != expected:
                raise AssertionError(f"K={k} {row['任务组编号']} 资源重算不一致：{actual} != {expected}")
            allocation = [allocation[i] + actual[i] for i in range(len(actual))]

            boxes = sum(box_sites[site] for site in sites)
            if boxes != int(row["货箱数"]) or len(selected) != int(row["运输架次数"]):
                raise AssertionError(f"K={k} {row['任务组编号']} 任务统计不一致")
            if len(relay_ids) != int(row["中继架次数"]):
                raise AssertionError(f"K={k} {row['任务组编号']} 中继统计不一致")
            transport_busy = sum(t["end"] - t["start"] for t in selected)
            relay_busy = sum(relays[r]["end"] - relays[r]["start"] for r in relay_ids)
            close(float(row["运输任务时长s"]), transport_busy)
            close(float(row["中继任务时长s"]), relay_busy)
            close(float(row["工作量s"]), transport_busy + relay_busy)
            close(float(row["运输能耗kWh"]), sum(t["energy"] for t in selected))
            close(float(row["中继能耗kWh"]), sum(relays[r]["energy"] for r in relay_ids))
            workloads.append(transport_busy + relay_busy)
            box_counts.append(boxes)

        compare = comparison[k]
        for name, value in zip(("A机", "B机", "C机", "A电池", "B电池", "C电池", "中继机", "中继组件"), allocation):
            if int(compare[f"配置_{name}"]) != value:
                raise AssertionError(f"K={k} 汇总资源 {name} 不一致")
        deficits = [max(0, allocation[i] - inventory[i]) for i in range(len(allocation))]
        spares = [max(0, inventory[i] - allocation[i]) for i in range(len(allocation))]
        for i, name in enumerate(("A机", "B机", "C机", "A电池", "B电池", "C电池", "中继机", "中继组件")):
            if int(compare[f"缺口_{name}"]) != deficits[i] or int(compare[f"冗余_{name}"]) != spares[i]:
                raise AssertionError(f"K={k} 的{name}缺口或冗余不一致")
        if int(compare["资源总缺口"]) != sum(deficits) or int(compare["资源总冗余"]) != sum(spares):
            raise AssertionError(f"K={k} 缺口或冗余汇总不一致")
        mean = sum(workloads) / k
        cv = math.sqrt(sum((value - mean) ** 2 for value in workloads) / k) / mean
        close(float(compare["工作量变异系数"]), cv)
        if int(compare["货箱数极差"]) != max(box_counts) - min(box_counts):
            raise AssertionError(f"K={k} 货箱数极差不一致")

    print("Q4 independent validation passed")
    print("checked: complete partition, inherited trips, relay dependencies, peak resources, inventory gaps")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Q4 validation failed: {exc}", file=sys.stderr)
        raise
