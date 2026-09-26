# -*- coding: utf-8 -*-
"""问题三：在 Q2 运输时间线上进行连续通信判定与中继候选点搜索。"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import math
import os
from dataclasses import dataclass

import numpy as np

from core import (GATEWAY_ANTENNA_AGL, Terrain, haversine_m, load_aircraft_types,
                  charge_time_s, load_boxes, load_dem, load_link_params, load_nodes,
                  load_relay_fleet, load_relay_type)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, "结果")


@dataclass(frozen=True)
class Point3D:
    lon: float
    lat: float
    alt_m: float


@dataclass(frozen=True)
class Phase:
    trip_id: str
    name: str
    start_s: float
    end_s: float
    p0: Point3D
    p1: Point3D

    def point(self, time_s: float) -> Point3D:
        if self.end_s <= self.start_s:
            return self.p1
        ratio = min(1.0, max(0.0, (time_s - self.start_s) / (self.end_s - self.start_s)))
        return Point3D(
            self.p0.lon + ratio * (self.p1.lon - self.p0.lon),
            self.p0.lat + ratio * (self.p1.lat - self.p0.lat),
            self.p0.alt_m + ratio * (self.p1.alt_m - self.p0.alt_m),
        )


@dataclass(frozen=True)
class CommSample:
    trip_id: str
    phase: str
    time_s: float
    point: Point3D
    direct: bool


@dataclass(frozen=True)
class RelayMission:
    mission_id: str
    drone_id: str
    component_id: str
    start_s: float
    hover: Point3D
    link_ready_s: float
    service_end_s: float
    return_s: float
    energy_kwh: float
    end_soc: float
    charge_complete_s: float


class LinkModel:
    def __init__(self):
        dem, lat, lon = load_dem()
        self.terrain = Terrain(dem, lat, lon)
        self.params = load_link_params()
        o = load_nodes()["O01"]
        self.gateway = Point3D(o.lon, o.lat, o.elevation_m + GATEWAY_ANTENNA_AGL)

    def obstructed(self, a: Point3D, b: Point3D) -> bool:
        horizontal = haversine_m(a.lon, a.lat, b.lon, b.lat)
        if horizontal < 1.0:
            return False
        di = abs(b.lat - a.lat) / abs(self.terrain.dlat)
        dj = abs(b.lon - a.lon) / abs(self.terrain.dlon)
        samples = max(3, int(math.ceil(max(di, dj) * 2.0)) + 1)
        lons, lats, ground = self.terrain.points_along(
            a.lon, a.lat, b.lon, b.lat, samples=samples
        )
        line_alt = np.linspace(a.alt_m, b.alt_m, samples)
        valid = ground > -32766.0
        valid[0] = False
        valid[-1] = False
        return bool(np.any(ground[valid] >= line_alt[valid] - 1e-8))

    def path_loss_db(self, a: Point3D, b: Point3D) -> float:
        horizontal = haversine_m(a.lon, a.lat, b.lon, b.lat)
        distance_km = math.hypot(horizontal, b.alt_m - a.alt_m) / 1000.0
        distance_km = max(distance_km, 1e-6)
        fspl = (32.45 + 20.0 * math.log10(self.params["frequency_mhz"])
                + 20.0 * math.log10(distance_km))
        return fspl + self.params["obstacle_loss_db"] * self.obstructed(a, b)

    def threshold_db(self, link: str) -> float:
        p = self.params
        receive_threshold = p["sensitivity_dbm"] + p["fade_margin_db"]
        if link == "direct":
            directions = (
                (p["transport_pt_dbm"], p["transport_gain_dbi"], p["gateway_gain_dbi"]),
                (p["gateway_pt_dbm"], p["gateway_gain_dbi"], p["transport_gain_dbi"]),
            )
        elif link == "access":
            directions = (
                (p["transport_pt_dbm"], p["transport_gain_dbi"], p["relay_access_gain_dbi"]),
                (p["relay_access_pt_dbm"], p["relay_access_gain_dbi"], p["transport_gain_dbi"]),
            )
        elif link == "backhaul":
            directions = (
                (p["relay_backhaul_pt_dbm"], p["relay_backhaul_gain_dbi"], p["gateway_gain_dbi"]),
                (p["gateway_pt_dbm"], p["gateway_gain_dbi"], p["relay_backhaul_gain_dbi"]),
            )
        else:
            raise ValueError(link)
        return min(pt + gt + gr - p["system_loss_db"] - receive_threshold
                   for pt, gt, gr in directions)

    def available(self, a: Point3D, b: Point3D, link: str) -> bool:
        return self.path_loss_db(a, b) <= self.threshold_db(link) + 1e-9


def load_q2_trips() -> list[dict]:
    path = os.path.join(RESULT_DIR, "Q2_运输架次.csv")
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["start_s"] = float(row["开始时刻s"])
        row["return_s"] = float(row["返回O01时刻s"])
        row["route"] = tuple(x for x in row["访问服务区顺序"].split(",") if x)
        row["box_ids"] = tuple(x for x in row["货箱编号列表"].split(",") if x)
    return rows


def transport_phases(trips: list[dict] | None = None) -> list[Phase]:
    aircraft = load_aircraft_types()
    nodes = load_nodes()
    boxes = {b["id"]: b for b in load_boxes()}
    dem, lat, lon = load_dem()
    terrain = Terrain(dem, lat, lon)
    phases: list[Phase] = []

    for trip in trips if trips is not None else load_q2_trips():
        ac = aircraft[trip["机型编号"]]
        counts = {stop: sum(boxes[x]["service"] == stop for x in trip["box_ids"])
                  for stop in trip["route"]}
        time_s = trip["start_s"] + ac.setup_time_s + ac.load_time_per_box_s * len(trip["box_ids"])
        src = "O01"
        for dst in trip["route"] + ("O01",):
            a, b = nodes[src], nodes[dst]
            ridge = terrain.ridge_elevation(a.lon, a.lat, b.lon, b.lat)
            cruise_alt = ridge + 50.0
            points = (
                Point3D(a.lon, a.lat, a.op_alt_m),
                Point3D(a.lon, a.lat, cruise_alt),
                Point3D(b.lon, b.lat, cruise_alt),
                Point3D(b.lon, b.lat, b.op_alt_m),
            )
            durations = (
                max(0.0, cruise_alt - a.op_alt_m) / ac.climb_speed_ms,
                haversine_m(a.lon, a.lat, b.lon, b.lat) / ac.cruise_speed_ms,
                max(0.0, cruise_alt - b.op_alt_m) / ac.descent_speed_ms,
            )
            for name, p0, p1, duration in zip(("爬升", "巡航", "下降"), points, points[1:], durations):
                phases.append(Phase(trip["架次编号"], name, time_s, time_s + duration, p0, p1))
                time_s += duration
            if dst != "O01":
                duration = ac.handover_base_s + ac.handover_per_box_s * counts[dst]
                phases.append(Phase(trip["架次编号"], "物资交接", time_s, time_s + duration,
                                    points[-1], points[-1]))
                time_s += duration
            src = dst
        if abs(time_s - trip["return_s"]) > 0.01:
            raise ValueError(f"{trip['架次编号']} 时间线与 Q2 返回时刻不一致：{time_s} vs {trip['return_s']}")
    return phases


def q3_transport_trips() -> list[dict]:
    """读取联合优化运输方案；无持久化方案时使用初始22架次调度。"""
    joint_path = os.path.join(RESULT_DIR, "Q3_transport_optimized.json")
    if os.path.exists(joint_path):
        with open(joint_path, encoding="utf-8") as f:
            trips = json.load(f)
        for trip in trips:
            trip["route"] = tuple(trip["route"])
            trip["box_ids"] = tuple(trip["box_ids"])
        return trips
    trips = copy.deepcopy(load_q2_trips())
    by_id = {t["架次编号"]: t for t in trips}
    early = by_id["Q2-014"]
    delayed = by_id["Q2-012"]
    duration_early = early["return_s"] - early["start_s"]
    duration_delayed = delayed["return_s"] - delayed["start_s"]

    early["start_s"] = by_id["Q2-009"]["return_s"]
    early["return_s"] = early["start_s"] + duration_early
    early["开始时刻s"] = f"{early['start_s']:.3f}"
    early["返回O01时刻s"] = f"{early['return_s']:.3f}"
    early["电池编号"] = "C-BAT-01"

    delayed["start_s"] = early["return_s"]
    delayed["return_s"] = delayed["start_s"] + duration_delayed
    delayed["开始时刻s"] = f"{delayed['start_s']:.3f}"
    delayed["返回O01时刻s"] = f"{delayed['return_s']:.3f}"
    delayed["电池编号"] = "C-BAT-04"
    return trips


def relay_profile(point: Point3D, link: LinkModel) -> dict:
    relay = load_relay_type()
    o = load_nodes()["O01"]
    ridge = link.terrain.ridge_elevation(o.lon, o.lat, point.lon, point.lat)
    cruise_alt = max(ridge + 50.0, point.alt_m)
    horizontal = haversine_m(o.lon, o.lat, point.lon, point.lat)
    outbound = ((cruise_alt - o.elevation_m) / relay["climb_speed_ms"]
                + horizontal / relay["cruise_speed_ms"]
                + (cruise_alt - point.alt_m) / relay["descent_speed_ms"])
    inbound = ((cruise_alt - point.alt_m) / relay["climb_speed_ms"]
               + horizontal / relay["cruise_speed_ms"]
               + (cruise_alt - o.elevation_m) / relay["descent_speed_ms"])
    climb_total = (cruise_alt - o.elevation_m) + (cruise_alt - point.alt_m)
    flight_energy = (2.0 * horizontal / relay["cruise_speed_ms"]
                     * relay["cruise_power_kw"] / 3600.0
                     + relay["takeoff_mass_kg"] * 9.80665 * climb_total
                     / relay["climb_eff"] / 3.6e6)
    link_energy = relay["hover_power_kw"] * relay["link_setup_s"] / 3600.0
    return {
        "outbound_s": outbound,
        "inbound_s": inbound,
        "ready_offset_s": relay["setup_time_s"] + outbound + relay["link_setup_s"],
        "base_energy_kwh": flight_energy + link_energy,
    }


def build_relay_missions(link: LinkModel) -> list[RelayMission]:
    relay = load_relay_type()
    _, _, full_charge_s = load_relay_fleet()
    definitions = (
        # 公共驻守点、早期东南点、后期北部点。
        ("Q3-R01", "R01", "R-EC-01", 109.2047137, 23.0470044, 300.0, 714.0, 5396.0),
        ("Q3-R02", "R02", "R-EC-02", 109.2692542, 23.0144974, 300.0, 714.0, 4094.0),
        ("Q3-R03", "R02", "R-EC-03", 109.2365633, 23.0626094, 150.0, 6522.0, 7511.0),
    )
    missions: list[RelayMission] = []
    config_path = os.path.join(RESULT_DIR, "Q3_relay_optimized.json")
    optimized = None
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            optimized = json.load(f)
        definitions = tuple((m["mission_id"], m["drone_id"], m["component_id"],
                             m["lon"], m["lat"], m["alt_m"], m["ready"], m["service_end"])
                            for m in optimized)
    for mission_id, drone, component, lon, lat, agl, ready, service_end in definitions:
        hover = Point3D(lon, lat, agl if optimized is not None else link.terrain.elevation(lon, lat) + agl)
        profile = relay_profile(hover, link)
        start = ready - profile["ready_offset_s"]
        service_duration = service_end - ready
        energy = (profile["base_energy_kwh"]
                  + (relay["hover_power_kw"] + relay["comm_extra_power_kw"])
                  * service_duration / 3600.0)
        end_soc = 1.0 - energy / relay["battery_kwh"]
        if end_soc < relay["reserve_ratio_pct"] / 100.0 - 1e-9:
            raise ValueError(f"{mission_id} 中继能源不足")
        return_s = service_end + profile["inbound_s"]
        charge_done = return_s + charge_time_s(full_charge_s, end_soc)
        missions.append(RelayMission(
            mission_id, drone, component, start, hover, ready, service_end,
            return_s, energy, end_soc, charge_done
        ))
    for drone in {m.drone_id for m in missions}:
        assigned = sorted((m for m in missions if m.drone_id == drone),key=lambda m:m.start_s)
        for previous,current in zip(assigned,assigned[1:]):
            if current.start_s < previous.return_s + relay["turnaround_s"] - 1e-7:
                raise ValueError(f"{drone} 中继架次周转时间不足")
    return missions


def coverage_source(point: Point3D, time_s: float, link: LinkModel,
                    missions: list[RelayMission]) -> tuple[str, str]:
    if link.available(point, link.gateway, "direct"):
        return "直连", ""
    for mission in missions:
        if (mission.link_ready_s - 1e-9 <= time_s <= mission.service_end_s + 1e-9
                and link.available(point, mission.hover, "access")):
            return "中继", mission.mission_id
    return "中断", ""


def verify_coverage(phases: list[Phase], link: LinkModel,
                    missions: list[RelayMission], step_s: float) -> tuple[int, int]:
    total = interrupted = 0
    for phase in phases:
        first_grid = math.ceil(phase.start_s / step_s) * step_s
        times = [phase.start_s, phase.end_s]
        times.extend(np.arange(first_grid, phase.end_s, step_s))
        for time_s in sorted({round(float(t), 9) for t in times}):
            total += 1
            status, _ = coverage_source(phase.point(time_s), time_s, link, missions)
            interrupted += status == "中断"
    return total, interrupted


def write_q3_results(trips: list[dict], phases: list[Phase], link: LinkModel,
                     missions: list[RelayMission]) -> None:
    os.makedirs(RESULT_DIR, exist_ok=True)
    source_path = os.path.join(ROOT, "结果", "Q2_运输架次.csv")
    with open(source_path, encoding="utf-8-sig", newline="") as f:
        fieldnames = csv.DictReader(f).fieldnames
    with open(os.path.join(RESULT_DIR, "Q3_运输架次.csv"), "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for trip in sorted(trips, key=lambda x: (x["start_s"], x["架次编号"])):
            writer.writerow({k: trip[k] for k in fieldnames})

    from q2 import hard_deadline
    boxes = {b["id"]: b for b in load_boxes()}
    delivery_fields = ["货箱编号", "架次编号", "服务区编号", "交付完成时刻s",
                       "硬时限s", "期望送达时刻s", "是否硬时限满足"]
    deliveries = []
    for trip in trips:
        handovers = [p.end_s for p in phases
                     if p.trip_id == trip["架次编号"] and p.name == "物资交接"]
        delivered_by_site = dict(zip(trip["route"], handovers))
        for bid in trip["box_ids"]:
            box = boxes[bid]
            delivered = delivered_by_site[box["service"]]
            deadline = hard_deadline(box)
            deliveries.append(dict(zip(delivery_fields, (
                bid, trip["架次编号"], box["service"], f"{delivered:.3f}",
                "" if deadline is None else deadline, box["expected_time_s"],
                "" if deadline is None else ("是" if delivered <= deadline + 1e-7 else "否")
            ))))
    with open(os.path.join(RESULT_DIR, "Q3_逐箱交付.csv"), "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=delivery_fields)
        writer.writeheader()
        writer.writerows(deliveries)

    relay_fields = ["中继架次编号", "中继无人机编号", "能源组件编号", "开始时刻s",
                    "悬停经度", "悬停纬度", "悬停海拔m", "建链完成时刻s",
                    "服务结束时刻s", "返回O01时刻s", "架次能耗kWh", "返航SOC%"]
    with open(os.path.join(RESULT_DIR, "Q3_中继架次.csv"), "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=relay_fields)
        writer.writeheader()
        for m in missions:
            writer.writerow(dict(zip(relay_fields, (
                m.mission_id, m.drone_id, m.component_id, f"{m.start_s:.3f}",
                f"{m.hover.lon:.7f}", f"{m.hover.lat:.7f}", f"{m.hover.alt_m:.3f}",
                f"{m.link_ready_s:.3f}", f"{m.service_end_s:.3f}", f"{m.return_s:.3f}",
                f"{m.energy_kwh:.6f}", f"{100.0*m.end_soc:.4f}"
            ))))

    comm_fields = ["运输架次编号", "通信阶段", "开始时刻s", "结束时刻s", "保障方式", "中继架次编号"]
    comm_rows = []
    for phase in phases:
        times = list(np.arange(phase.start_s, phase.end_s, 1.0)) + [phase.end_s]
        states = [(float(t),) + coverage_source(phase.point(float(t)), float(t), link, missions)
                  for t in times]
        run_start, way, relay_id = states[0]
        for index in range(1, len(states)):
            time_s, next_way, next_relay = states[index]
            if (next_way, next_relay) != (way, relay_id):
                comm_rows.append((phase.trip_id, phase.name, run_start, time_s, way, relay_id))
                run_start, way, relay_id = time_s, next_way, next_relay
        comm_rows.append((phase.trip_id, phase.name, run_start, phase.end_s, way, relay_id))
    with open(os.path.join(RESULT_DIR, "Q3_通信保障.csv"), "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(comm_fields)
        for row in comm_rows:
            writer.writerow([row[0], row[1], f"{row[2]:.3f}", f"{row[3]:.3f}", row[4], row[5]])

    relay_energy = sum(m.energy_kwh for m in missions)
    joint_makespan = max(max(t["return_s"] for t in trips), max(m.return_s for m in missions))
    metrics = (
        ("运输架次数", len(trips)), ("中继架次数", len(missions)),
        ("硬时限违约箱数", sum(row["是否硬时限满足"] == "否" for row in deliveries)),
        ("普通物资迟到箱数", sum(not row["硬时限s"] and float(row["交付完成时刻s"]) > float(row["期望送达时刻s"]) + 1e-7 for row in deliveries)),
        ("运输能耗kWh", sum(float(t["架次能耗kWh"]) for t in trips)),
        ("中继能耗kWh", relay_energy),
        ("总能耗kWh", sum(float(t["架次能耗kWh"]) for t in trips) + relay_energy),
        ("联合任务完成时间s", joint_makespan),
    )
    with open(os.path.join(RESULT_DIR, "Q3_指标汇总.csv"), "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["指标", "数值"])
        writer.writerows(metrics)


def solve_q3() -> None:
    link = LinkModel()
    trips = q3_transport_trips()
    phases = transport_phases(trips)
    missions = build_relay_missions(link)
    for step in (5.0, 1.0, 0.5):
        total, interrupted = verify_coverage(phases, link, missions, step)
        print(f"通信复核 step={step:g}s: samples={total}, interruptions={interrupted}")
        if interrupted:
            raise RuntimeError(f"{step:g}s 粒度发现通信中断")
    write_q3_results(trips, phases, link, missions)
    print(f"Q3: transport sorties={len(trips)}, relay sorties={len(missions)}, "
          f"joint makespan={max(max(t['return_s'] for t in trips), max(m.return_s for m in missions)):.1f}s")
    print(f"relay energy={sum(m.energy_kwh for m in missions):.4f} kWh")


def sample_phases(phases: list[Phase], link: LinkModel, step_s: float) -> list[CommSample]:
    nodes = load_nodes()
    o = nodes["O01"]
    gateway = Point3D(o.lon, o.lat, o.elevation_m + GATEWAY_ANTENNA_AGL)
    samples: list[CommSample] = []
    for phase in phases:
        first_grid = math.ceil(phase.start_s / step_s) * step_s
        times = [phase.start_s, phase.end_s]
        times.extend(np.arange(first_grid, phase.end_s, step_s))
        times = sorted({round(float(t), 9) for t in times})
        for time_s in times:
            point = phase.point(float(time_s))
            samples.append(CommSample(
                phase.trip_id, phase.name, float(time_s), point,
                link.available(point, gateway, "direct")
            ))
    return samples


def candidate_points(gaps: list[CommSample], link: LinkModel) -> list[Point3D]:
    nodes = load_nodes()
    o = nodes["O01"]
    horizontal: list[tuple[float, float]] = [(o.lon, o.lat)]
    for node in nodes.values():
        horizontal.append((node.lon, node.lat))
        horizontal.append(((o.lon + node.lon) / 2.0, (o.lat + node.lat) / 2.0))
    stride = max(1, len(gaps) // 80)
    horizontal.extend((s.point.lon, s.point.lat) for s in gaps[::stride])

    candidates: list[Point3D] = []
    seen = set()
    for lon, lat in horizontal:
        ground = link.terrain.elevation(lon, lat)
        for agl in (150.0, 225.0, 300.0):
            key = (round(lon, 6), round(lat, 6), agl)
            if key not in seen:
                seen.add(key)
                candidates.append(Point3D(lon, lat, ground + agl))
    return candidates


def analyze(step_s: float) -> None:
    relay = load_relay_type()
    relay_drones, relay_components, relay_charge_s = load_relay_fleet()
    link = LinkModel()
    phases = transport_phases()
    samples = sample_phases(phases, link, step_s)
    gaps = [s for s in samples if not s.direct]
    print(f"运输通信采样点={len(samples)}, 直连中断点={len(gaps)}, 中断率={len(gaps)/len(samples):.2%}")
    print(f"链路门限: direct={link.threshold_db('direct'):.1f} dB, "
          f"access={link.threshold_db('access'):.1f} dB, "
          f"backhaul={link.threshold_db('backhaul'):.1f} dB")
    print(f"中继资源: 无人机={relay_drones}, 能源组件={relay_components}, "
          f"完全充电={relay_charge_s:.0f}s, 最大悬停={relay['max_hover_agl_m']:.0f}m")
    if not gaps:
        return
    nodes = load_nodes()
    o = nodes["O01"]
    gateway = Point3D(o.lon, o.lat, o.elevation_m + GATEWAY_ANTENNA_AGL)
    scored = []
    for candidate in candidate_points(gaps, link):
        if not link.available(candidate, gateway, "backhaul"):
            continue
        covered = sum(link.available(s.point, candidate, "access") for s in gaps)
        scored.append((covered, candidate))
    scored.sort(key=lambda x: x[0], reverse=True)
    print(f"可回传候选点={len(scored)}")
    for covered, point in scored[:10]:
        ground = link.terrain.elevation(point.lon, point.lat)
        print(f"  cover={covered}/{len(gaps)} ({covered/len(gaps):.2%}) "
              f"lon={point.lon:.7f} lat={point.lat:.7f} "
              f"alt={point.alt_m:.1f}m agl={point.alt_m-ground:.1f}m")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=float, default=10.0)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.analyze_only:
        analyze(args.step)
    else:
        solve_q3()


if __name__ == "__main__":
    main()
