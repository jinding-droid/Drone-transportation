# -*- coding: utf-8 -*-
"""独立复核问题二输出：物理量、时限、无人机和电池时间线。"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

from core import (Terrain, build_all_segments, charge_time_s, load_aircraft_types,
                  load_boxes, load_dem, load_fleet, load_nodes, segment_energy_kwh)
from q2 import hard_deadline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, "结果")


def read_csv(name: str) -> list[dict]:
    with open(os.path.join(RESULT_DIR, name), encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main(suffix: str = "", require_zero_late: bool = False) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    aircraft = load_aircraft_types()
    boxes = load_boxes()
    box_by_id = {b["id"]: b for b in boxes}
    drones, battery_stock, full_charge = load_fleet()
    drone_type = dict(drones)
    nodes = load_nodes()
    dem, lat, lon = load_dem()
    segments = build_all_segments(Terrain(dem, lat, lon), nodes)

    tag = f"_{suffix}" if suffix else ""
    trip_rows = read_csv(f"Q2_运输架次{tag}.csv")
    delivery_rows = read_csv(f"Q2_逐箱交付{tag}.csv")
    delivery_by_box = {r["货箱编号"]: r for r in delivery_rows}
    assert len(delivery_by_box) == len(delivery_rows) == len(boxes)

    seen = []
    drone_intervals = defaultdict(list)
    battery_intervals = defaultdict(list)
    totals = {"energy": 0.0, "max_return": 0.0}
    soft_late_count = 0

    for row in trip_rows:
        trip_id = row["架次编号"]
        g = row["机型编号"]
        ac = aircraft[g]
        drone = row["无人机编号"]
        battery = row["电池编号"]
        assert drone_type[drone] == g, (trip_id, drone, g)
        assert battery.startswith(f"{g}-BAT-"), (trip_id, battery, g)
        battery_index = int(battery.rsplit("-", 1)[1])
        assert 1 <= battery_index <= battery_stock[g]

        start = float(row["开始时刻s"])
        order = tuple(x for x in row["访问服务区顺序"].split(",") if x)
        box_ids = tuple(x for x in row["货箱编号列表"].split(",") if x)
        trip_boxes = [box_by_id[x] for x in box_ids]
        seen.extend(box_ids)
        assert set(order) == {b["service"] for b in trip_boxes}, trip_id

        mass = sum(b["mass_kg"] for b in trip_boxes)
        volume = sum(b["volume_m3"] for b in trip_boxes)
        assert mass <= ac.max_payload_kg + 1e-9, (trip_id, mass)
        assert volume <= ac.volume_m3 + 1e-12, (trip_id, volume)

        elapsed = ac.setup_time_s + len(trip_boxes) * ac.load_time_per_box_s
        energy = 0.0
        payload = mass
        src = "O01"
        for stop in order:
            seg = segments[(src, stop)]
            energy += segment_energy_kwh(ac, payload, seg)
            elapsed += seg.flight_time_s(ac)
            stop_boxes = [b for b in trip_boxes if b["service"] == stop]
            elapsed += ac.handover_base_s + ac.handover_per_box_s * len(stop_boxes)
            for b in stop_boxes:
                delivered = start + elapsed
                reported = float(delivery_by_box[b["id"]]["交付完成时刻s"])
                assert abs(delivered - reported) <= 0.11, (b["id"], delivered, reported)
                deadline = hard_deadline(b)
                if deadline is not None:
                    assert delivered <= deadline + 1e-7, (b["id"], delivered, deadline)
                if delivered > b["expected_time_s"] + 1e-7:
                    soft_late_count += 1
            payload -= sum(b["mass_kg"] for b in stop_boxes)
            src = stop
        back = segments[(src, "O01")]
        energy += segment_energy_kwh(ac, 0.0, back)
        elapsed += back.flight_time_s(ac)
        returned = start + elapsed

        reported_energy = float(row["架次能耗kWh"])
        reported_return = float(row["返回O01时刻s"])
        assert abs(energy - reported_energy) <= 5.1e-5, (trip_id, energy, reported_energy)
        assert abs(returned - reported_return) <= 0.11, (trip_id, returned, reported_return)
        assert energy <= (1.0 - ac.reserve_ratio) * ac.battery_kwh + 1e-9

        soc = 1.0 - energy / ac.battery_kwh
        charge_done = returned + charge_time_s(full_charge[g], soc)
        drone_intervals[drone].append((start, returned, trip_id))
        battery_intervals[battery].append((start, charge_done, trip_id))
        totals["energy"] += energy
        totals["max_return"] = max(totals["max_return"], returned)

    assert len(seen) == len(set(seen)) == len(boxes)
    assert set(seen) == set(box_by_id)
    if require_zero_late:
        assert soft_late_count == 0, soft_late_count

    for resource, intervals in list(drone_intervals.items()) + list(battery_intervals.items()):
        intervals.sort()
        for prev, cur in zip(intervals, intervals[1:]):
            assert cur[0] >= prev[1] - 1.1e-3, (resource, prev, cur)

    print("Q2 validation passed")
    print(f"sorties={len(trip_rows)}, boxes={len(seen)}")
    print(f"energy={totals['energy']:.4f} kWh, makespan={totals['max_return']:.1f} s")
    print("hard deadlines: all satisfied")
    print(f"expected-time late boxes: {soft_late_count}")
    print("drone and battery timelines: no overlap; batteries fully recharged before reuse")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('suffix', nargs='?', default='')
    parser.add_argument('--zero-late', action='store_true')
    args = parser.parse_args()
    main(args.suffix, args.zero_late)
