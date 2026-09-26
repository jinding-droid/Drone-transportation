# -*- coding: utf-8 -*-
"""独立复核问题一结果文件的覆盖性、物理可行性与架次数下界。"""

from __future__ import annotations

import csv
import math
import os
import sys
from collections import Counter, defaultdict

from core import (Terrain, build_segment, load_aircraft_types, load_boxes,
                  load_dem, load_nodes)
from q1 import round_trip_energy, round_trip_time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAN = os.path.join(ROOT, "结果", "Q1_组批方案.csv")
TOL = 1e-4


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    acs = load_aircraft_types()
    nodes = load_nodes()
    boxes = load_boxes()
    box_by_id = {box["id"]: box for box in boxes}
    dem, lat, lon = load_dem()
    terrain = Terrain(dem, lat, lon)
    segments = {
        service: (
            build_segment(terrain, nodes["O01"], nodes[service]),
            build_segment(terrain, nodes[service], nodes["O01"]),
        )
        for service in nodes if service != "O01"
    }

    with open(PLAN, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    assigned = []
    errors = []
    for row in rows:
        trip = row["架次编号"]
        service = row["服务区编号"]
        ac = acs[row["机型编号"]]
        ids = [item for item in row["货箱编号列表"].split(",") if item]
        assigned.extend(ids)
        unknown = [item for item in ids if item not in box_by_id]
        if unknown:
            errors.append(f"{trip}: 未知货箱 {unknown}")
            continue
        trip_boxes = [box_by_id[item] for item in ids]
        if any(box["service"] != service for box in trip_boxes):
            errors.append(f"{trip}: 存在跨服务区货箱")
        mass = sum(box["mass_kg"] for box in trip_boxes)
        volume = sum(box["volume_m3"] for box in trip_boxes)
        energy = round_trip_energy(ac, *segments[service], mass)
        duration = round_trip_time(ac, *segments[service], len(ids))
        soc = 100.0 * (1.0 - energy / ac.battery_kwh)
        checks = [
            (mass <= ac.max_payload_kg + TOL, "质量超限"),
            (volume <= ac.volume_m3 + TOL, "体积超限"),
            (energy <= (1.0 - ac.reserve_ratio) * ac.battery_kwh + TOL, "能量超限"),
            (abs(mass - float(row["总质量kg"])) <= TOL, "质量字段不一致"),
            (abs(volume - float(row["总体积m³"])) <= TOL, "体积字段不一致"),
            (abs(energy - float(row["架次能耗kWh"])) <= TOL, "能耗字段不一致"),
            (abs(duration - float(row["往返时间s"])) <= 0.11, "时间字段不一致"),
            (abs(soc - float(row["返航SOC%"])) <= 0.011, "SOC字段不一致"),
        ]
        errors.extend(f"{trip}: {message}" for ok, message in checks if not ok)

    counts = Counter(assigned)
    missing = sorted(set(box_by_id) - set(assigned))
    duplicates = sorted(item for item, count in counts.items() if count != 1)
    if missing:
        errors.append(f"遗漏货箱: {missing}")
    if duplicates:
        errors.append(f"重复货箱: {duplicates}")

    by_service = defaultdict(list)
    for box in boxes:
        by_service[box["service"]].append(box)
    lower_bound = 0
    for service_boxes in by_service.values():
        mass_lb = math.ceil(sum(box["mass_kg"] for box in service_boxes) / max(ac.max_payload_kg for ac in acs.values()))
        volume_lb = math.ceil(sum(box["volume_m3"] for box in service_boxes) / max(ac.volume_m3 for ac in acs.values()))
        lower_bound += max(1, mass_lb, volume_lb)

    if len(rows) != lower_bound:
        errors.append(f"架次数 {len(rows)} 未达到可证明下界 {lower_bound}")
    if errors:
        print("Q1 校验失败：")
        for error in errors:
            print(" -", error)
        raise SystemExit(1)

    print("Q1 校验通过")
    print("架次数:", len(rows), "| 可证明下界:", lower_bound)
    print("货箱:", len(assigned), "| 总质量: %.1f kg | 总体积: %.3f m³"
          % (sum(box["mass_kg"] for box in boxes), sum(box["volume_m3"] for box in boxes)))
    print("总能耗: %.4f kWh | 累计作业时间: %.1f s"
          % (sum(float(row["架次能耗kWh"]) for row in rows),
             sum(float(row["往返时间s"]) for row in rows)))
    print("机型架次:", dict(sorted(Counter(row["机型编号"] for row in rows).items())))


if __name__ == "__main__":
    main()
