# -*- coding: utf-8 -*-
"""问题四：冻结 Q3 调度，对 15 个服务区作 2/3 组独立资源分区。"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from functools import lru_cache

from core import (charge_time_s, haversine_m, load_aircraft_types, load_boxes,
                  load_fleet, load_nodes, load_relay_fleet, load_relay_type)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, "结果")
SITES = tuple(f"S{i:03d}" for i in range(1, 16))
SITE_INDEX = {site: i for i, site in enumerate(SITES)}
BALANCE_CV_LIMIT = 0.10


def read_csv(name: str) -> list[dict]:
    with open(os.path.join(RESULT_DIR, name), encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def peak(intervals: list[tuple[float, float]]) -> int:
    events = []
    for start, end in intervals:
        # Q3 提交表的时刻精确到 0.001 s；统一精度可消除由能耗反算充电
        # 时长时产生的 1e-4 s 级伪重叠。
        events.append((round(start, 3), 1))
        events.append((round(end, 3), -1))
    active = answer = 0
    for _, delta in sorted(events, key=lambda x: (x[0], x[1])):
        active += delta
        answer = max(answer, active)
    return answer


@dataclass(frozen=True)
class GroupMetrics:
    mask: int
    sites: tuple[str, ...]
    boxes: int
    transport_sorties: int
    relay_sorties: int
    transport_energy: float
    relay_energy: float
    transport_busy_s: float
    relay_busy_s: float
    resources: tuple[int, ...]

    @property
    def workload(self) -> float:
        # 运输和中继任务从开始至返航的总占用时长，作为组间可比的综合工作量。
        return self.transport_busy_s + self.relay_busy_s


class Q4Model:
    resource_names = ("A机", "B机", "C机", "A电池", "B电池", "C电池", "中继机", "中继组件")

    def __init__(self):
        self.aircraft = load_aircraft_types()
        self.nodes = load_nodes()
        self.boxes = load_boxes()
        drones, battery_stock, full_charge = load_fleet()
        relay_drones, relay_components, relay_full_charge = load_relay_fleet()
        drone_stock = {g: sum(drone_type == g for _, drone_type in drones) for g in "ABC"}
        self.inventory = (drone_stock["A"], drone_stock["B"], drone_stock["C"],
                          battery_stock["A"], battery_stock["B"], battery_stock["C"],
                          len(relay_drones), relay_components)

        self.trips = []
        for row in read_csv("Q3_运输架次.csv"):
            sites = tuple(x for x in row["访问服务区顺序"].split(",") if x)
            g = row["机型编号"]
            start, end = float(row["开始时刻s"]), float(row["返回O01时刻s"])
            energy = float(row["架次能耗kWh"])
            soc = 1.0 - energy / self.aircraft[g].battery_kwh
            self.trips.append({
                "id": row["架次编号"], "sites": sites, "type": g, "start": start, "end": end,
                "energy": energy, "battery_end": end + charge_time_s(full_charge[g], soc),
                "boxes": tuple(x for x in row["货箱编号列表"].split(",") if x),
            })

        relay = load_relay_type()
        self.relays = {}
        for row in read_csv("Q3_中继架次.csv"):
            start, end = float(row["开始时刻s"]), float(row["返回O01时刻s"])
            energy = float(row["架次能耗kWh"])
            soc = 1.0 - energy / relay["battery_kwh"]
            self.relays[row["中继架次编号"]] = {
                "start": start, "end": end, "drone_end": end + relay["turnaround_s"],
                "component_end": end + charge_time_s(relay_full_charge, soc), "energy": energy,
            }

        self.trip_relays: dict[str, set[str]] = {t["id"]: set() for t in self.trips}
        for row in read_csv("Q3_通信保障.csv"):
            if row["保障方式"] == "中继" and row["中继架次编号"]:
                self.trip_relays[row["运输架次编号"]].add(row["中继架次编号"])

        self.box_count = {site: sum(b["service"] == site for b in self.boxes) for site in SITES}
        self.components = self._components()
        self.global_metrics = self.metrics((1 << len(SITES)) - 1)

    def _components(self) -> tuple[int, ...]:
        parent = list(range(len(SITES)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            a, b = find(a), find(b)
            if a != b:
                parent[b] = a

        for trip in self.trips:
            ids = [SITE_INDEX[s] for s in trip["sites"]]
            for index in ids[1:]:
                union(ids[0], index)
        masks: dict[int, int] = {}
        for index in range(len(SITES)):
            root = find(index)
            masks[root] = masks.get(root, 0) | (1 << index)
        return tuple(sorted(masks.values(), key=lambda m: (m & -m).bit_length()))

    @lru_cache(maxsize=None)
    def metrics(self, mask: int) -> GroupMetrics:
        sites = tuple(site for site in SITES if mask & (1 << SITE_INDEX[site]))
        selected = [t for t in self.trips if any(mask & (1 << SITE_INDEX[s]) for s in t["sites"])]
        if any(not all(mask & (1 << SITE_INDEX[s]) for s in t["sites"]) for t in selected):
            raise ValueError("多服务区架次被拆分")
        relay_ids = sorted(set().union(*(self.trip_relays[t["id"]] for t in selected))) if selected else []
        drone_counts = []
        battery_counts = []
        for g in "ABC":
            typed = [t for t in selected if t["type"] == g]
            drone_counts.append(peak([(t["start"], t["end"]) for t in typed]))
            battery_counts.append(peak([(t["start"], t["battery_end"]) for t in typed]))
        relay_drone_count = peak([(self.relays[r]["start"], self.relays[r]["drone_end"]) for r in relay_ids])
        relay_component_count = peak([(self.relays[r]["start"], self.relays[r]["component_end"]) for r in relay_ids])
        resources = tuple(drone_counts + battery_counts + [relay_drone_count, relay_component_count])
        return GroupMetrics(
            mask, sites, sum(self.box_count[s] for s in sites), len(selected), len(relay_ids),
            sum(t["energy"] for t in selected), sum(self.relays[r]["energy"] for r in relay_ids),
            sum(t["end"] - t["start"] for t in selected),
            sum(self.relays[r]["end"] - self.relays[r]["start"] for r in relay_ids), resources,
        )

    def spatial_cost(self, groups: tuple[GroupMetrics, ...]) -> float:
        cost = 0.0
        for group in groups:
            lon = sum(self.nodes[s].lon for s in group.sites) / len(group.sites)
            lat = sum(self.nodes[s].lat for s in group.sites) / len(group.sites)
            cost += sum(haversine_m(lon, lat, self.nodes[s].lon, self.nodes[s].lat) for s in group.sites)
        return cost

    def objective(self, masks: tuple[int, ...]) -> tuple:
        groups = tuple(self.metrics(mask) for mask in masks)
        allocation = tuple(sum(g.resources[i] for g in groups) for i in range(len(self.resource_names)))
        deficits = tuple(max(0, allocation[i] - self.inventory[i]) for i in range(len(allocation)))
        total_deficit = sum(deficits)
        duplication = sum(allocation) - sum(self.global_metrics.resources)
        workloads = [g.workload for g in groups]
        mean = sum(workloads) / len(workloads)
        workload_cv = math.sqrt(sum((x - mean) ** 2 for x in workloads) / len(workloads)) / mean
        box_range = max(g.boxes for g in groups) - min(g.boxes for g in groups)
        balance_violation = max(0.0, workload_cv - BALANCE_CV_LIMIT)
        return (round(balance_violation, 12), total_deficit, round(workload_cv, 12),
                box_range, duplication,
                round(self.spatial_cost(groups), 6), tuple(sorted(masks)))

    def solve(self, k: int) -> tuple[GroupMetrics, ...]:
        best_objective = None
        best_masks = None
        components = self.components
        assignment = [0] * len(components)

        def visit(index: int, maximum: int) -> None:
            nonlocal best_objective, best_masks
            if index == len(components):
                if maximum != k - 1:
                    return
                masks = [0] * k
                for component, group in zip(components, assignment):
                    masks[group] |= component
                candidate = tuple(masks)
                objective = self.objective(candidate)
                if best_objective is None or objective < best_objective:
                    best_objective, best_masks = objective, candidate
                return
            for group in range(min(maximum + 1, k - 1) + 1):
                assignment[index] = group
                visit(index + 1, max(maximum, group))

        assignment[0] = 0
        visit(1, 0)
        assert best_masks is not None
        return tuple(self.metrics(mask) for mask in best_masks)


def write_results(model: Q4Model, solutions: dict[int, tuple[GroupMetrics, ...]]) -> None:
    rows = []
    comparison = []
    for k, groups in solutions.items():
        allocation = tuple(sum(g.resources[i] for g in groups) for i in range(len(model.resource_names)))
        deficit = tuple(max(0, allocation[i] - model.inventory[i]) for i in range(len(allocation)))
        spare = tuple(max(0, model.inventory[i] - allocation[i]) for i in range(len(allocation)))
        workloads = [g.workload for g in groups]
        mean = sum(workloads) / len(workloads)
        cv = math.sqrt(sum((x - mean) ** 2 for x in workloads) / len(workloads)) / mean
        for index, group in enumerate(groups, 1):
            rows.append((k, f"G{index}", ",".join(group.sites), *group.resources,
                         group.boxes, group.transport_sorties, group.relay_sorties,
                         group.transport_energy, group.relay_energy, group.transport_busy_s,
                         group.relay_busy_s, group.workload))
        comparison.append((k, *allocation, *deficit, *spare, cv,
                           max(g.boxes for g in groups) - min(g.boxes for g in groups),
                           sum(deficit), sum(spare)))

    header = ["K", "任务组编号", "服务区列表", "A型运输无人机数", "B型运输无人机数",
              "C型运输无人机数", "A型电池组数", "B型电池组数", "C型电池组数",
              "中继无人机数", "中继能源组件数", "货箱数", "运输架次数", "中继架次数",
              "运输能耗kWh", "中继能耗kWh", "运输任务时长s", "中继任务时长s", "工作量s"]
    with open(os.path.join(RESULT_DIR, "Q4_分区配置.csv"), "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    resource_labels = list(model.resource_names)
    compare_header = (["K"] + [f"配置_{x}" for x in resource_labels]
                      + [f"缺口_{x}" for x in resource_labels]
                      + [f"冗余_{x}" for x in resource_labels]
                      + ["工作量变异系数", "货箱数极差", "资源总缺口", "资源总冗余"])
    with open(os.path.join(RESULT_DIR, "Q4_方案比较.csv"), "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(compare_header)
        writer.writerows(comparison)


def main() -> None:
    model = Q4Model()
    solutions = {k: model.solve(k) for k in (2, 3)}
    write_results(model, solutions)
    print("Q4 partition solved")
    print("inventory:", dict(zip(model.resource_names, model.inventory)))
    for k, groups in solutions.items():
        allocation = tuple(sum(g.resources[i] for g in groups) for i in range(len(model.resource_names)))
        deficit = tuple(max(0, allocation[i] - model.inventory[i]) for i in range(len(allocation)))
        print(f"K={k}, allocation={dict(zip(model.resource_names, allocation))}, "
              f"deficit={dict(zip(model.resource_names, deficit))}")
        for index, group in enumerate(groups, 1):
            print(f"  G{index}: sites={','.join(group.sites)}, boxes={group.boxes}, "
                  f"trips={group.transport_sorties}+{group.relay_sorties}, resources={group.resources}")


if __name__ == "__main__":
    main()
