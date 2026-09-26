# -*- coding: utf-8 -*-
"""独立复核 Q3 的运输、时限、中继能源、资源周转与连续通信。"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

from core import (Terrain, build_all_segments, charge_time_s, load_aircraft_types,
                  load_boxes, load_dem, load_fleet, load_nodes, load_relay_fleet,
                  load_relay_type, segment_energy_kwh)
from q2 import hard_deadline
from q3 import (LinkModel, Point3D, RelayMission, relay_profile, transport_phases,
                verify_coverage)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, "结果")


def read_csv(name: str) -> list[dict]:
    with open(os.path.join(RESULT_DIR, name), encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    aircraft = load_aircraft_types()
    boxes = load_boxes()
    box_by_id = {b["id"]: b for b in boxes}
    drones, battery_stock, full_charge = load_fleet()
    drone_type = dict(drones)
    nodes = load_nodes()
    dem, lat, lon = load_dem()
    segments = build_all_segments(Terrain(dem, lat, lon), nodes)
    trip_rows = read_csv("Q3_运输架次.csv")
    delivery_rows = read_csv("Q3_逐箱交付.csv")
    delivery_by_box = {r["货箱编号"]: r for r in delivery_rows}
    assert len(delivery_by_box) == len(delivery_rows) == len(boxes)

    seen: list[str] = []
    drone_intervals = defaultdict(list)
    battery_intervals = defaultdict(list)
    transport_energy = 0.0
    parsed_trips = []
    for row in trip_rows:
        trip_id = row["架次编号"]
        g = row["机型编号"]
        ac = aircraft[g]
        drone, battery = row["无人机编号"], row["电池编号"]
        assert drone_type[drone] == g
        assert battery.startswith(f"{g}-BAT-")
        assert 1 <= int(battery.rsplit("-", 1)[1]) <= battery_stock[g]
        start = float(row["开始时刻s"])
        assert start >= 0
        order = tuple(x for x in row["访问服务区顺序"].split(",") if x)
        box_ids = tuple(x for x in row["货箱编号列表"].split(",") if x)
        trip_boxes = [box_by_id[x] for x in box_ids]
        seen.extend(box_ids)
        assert set(order) == {b["service"] for b in trip_boxes}
        mass = sum(b["mass_kg"] for b in trip_boxes)
        volume = sum(b["volume_m3"] for b in trip_boxes)
        assert mass <= ac.max_payload_kg + 1e-9
        assert volume <= ac.volume_m3 + 1e-12

        elapsed = ac.setup_time_s + ac.load_time_per_box_s * len(box_ids)
        energy = 0.0
        payload = mass
        src = "O01"
        for stop in order:
            seg = segments[(src, stop)]
            energy += segment_energy_kwh(ac, payload, seg)
            elapsed += seg.flight_time_s(ac)
            stop_boxes = [b for b in trip_boxes if b["service"] == stop]
            elapsed += ac.handover_base_s + ac.handover_per_box_s * len(stop_boxes)
            for box in stop_boxes:
                delivered = start + elapsed
                reported = float(delivery_by_box[box["id"]]["交付完成时刻s"])
                assert abs(delivered - reported) <= 0.11
                deadline = hard_deadline(box)
                if deadline is not None:
                    assert delivered <= deadline + 1e-7, (box["id"], delivered, deadline)
                assert delivered <= box["expected_time_s"] + 1e-7
            payload -= sum(b["mass_kg"] for b in stop_boxes)
            src = stop
        back = segments[(src, "O01")]
        energy += segment_energy_kwh(ac, 0.0, back)
        elapsed += back.flight_time_s(ac)
        returned = start + elapsed
        assert abs(returned - float(row["返回O01时刻s"])) <= 0.11
        assert abs(energy - float(row["架次能耗kWh"])) <= 5.1e-5
        assert energy <= (1.0 - ac.reserve_ratio) * ac.battery_kwh + 1e-9
        charge_done = returned + charge_time_s(full_charge[g], 1.0 - energy / ac.battery_kwh)
        drone_intervals[drone].append((start, returned, trip_id))
        battery_intervals[battery].append((start, charge_done, trip_id))
        transport_energy += energy
        parsed = dict(row)
        parsed.update(start_s=start, return_s=returned, route=order, box_ids=box_ids)
        parsed_trips.append(parsed)

    assert len(seen) == len(set(seen)) == len(boxes)
    for resource, intervals in list(drone_intervals.items()) + list(battery_intervals.items()):
        intervals.sort()
        for previous, current in zip(intervals, intervals[1:]):
            assert current[0] >= previous[1] - 1.1e-3, (resource, previous, current)

    link = LinkModel()
    relay = load_relay_type()
    relay_drones, component_stock, relay_full_charge = load_relay_fleet()
    relay_rows = read_csv("Q3_中继架次.csv")
    relay_intervals = defaultdict(list)
    component_intervals = defaultdict(list)
    missions: list[RelayMission] = []
    relay_energy = 0.0
    for row in relay_rows:
        hover = Point3D(float(row["悬停经度"]), float(row["悬停纬度"]), float(row["悬停海拔m"]))
        ground = link.terrain.elevation(hover.lon, hover.lat)
        assert -1e-7 <= hover.alt_m - ground <= relay["max_hover_agl_m"] + 1e-7
        assert link.available(hover, link.gateway, "backhaul")
        profile = relay_profile(hover, link)
        start = float(row["开始时刻s"])
        ready = float(row["建链完成时刻s"])
        service_end = float(row["服务结束时刻s"])
        assert 0 <= start <= ready <= service_end
        returned = float(row["返回O01时刻s"])
        energy = float(row["架次能耗kWh"])
        expected_energy = (profile["base_energy_kwh"]
                           + (relay["hover_power_kw"] + relay["comm_extra_power_kw"])
                           * (service_end - ready) / 3600.0)
        assert abs(ready - (start + profile["ready_offset_s"])) <= 0.01
        assert abs(returned - (service_end + profile["inbound_s"])) <= 0.01
        assert abs(energy - expected_energy) <= 5.1e-6
        assert energy <= (1.0 - relay["reserve_ratio_pct"] / 100.0) * relay["battery_kwh"] + 1e-9
        soc = 1.0 - energy / relay["battery_kwh"]
        charge_done = returned + charge_time_s(relay_full_charge, soc)
        drone = row["中继无人机编号"]
        component = row["能源组件编号"]
        assert drone in relay_drones
        assert 1 <= int(component.rsplit("-", 1)[1]) <= component_stock
        relay_intervals[drone].append((start, returned + relay["turnaround_s"], row["中继架次编号"]))
        component_intervals[component].append((start, charge_done, row["中继架次编号"]))
        missions.append(RelayMission(row["中继架次编号"], drone, component, start, hover,
                                     ready, service_end, returned, energy, soc, charge_done))
        relay_energy += energy

    for resource, intervals in list(relay_intervals.items()) + list(component_intervals.items()):
        intervals.sort()
        for previous, current in zip(intervals, intervals[1:]):
            assert current[0] >= previous[1] - 1.1e-3, (resource, previous, current)

    phases = transport_phases(parsed_trips)
    samples, interrupted = verify_coverage(phases, link, missions, 0.5)
    assert interrupted == 0
    joint_makespan = max(max(t["return_s"] for t in parsed_trips), max(m.return_s for m in missions))
    print("Q3 validation passed")
    print(f"transport sorties={len(trip_rows)}, relay sorties={len(relay_rows)}, boxes={len(seen)}")
    print(f"energy={transport_energy + relay_energy:.4f} kWh, joint makespan={joint_makespan:.1f} s")
    print(f"sampled communication: {samples} samples at 0.5 s, interruptions=0")
    print("transport and relay resource timelines: no overlap")


if __name__ == "__main__":
    main()
