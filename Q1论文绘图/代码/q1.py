# -*- coding: utf-8 -*-
"""问题一：单点往返运输能力与货箱组批（精确高效实现）。

关键观察：全部 80 个货箱只有 4 种（质量, 体积）组合：
    (3 kg, 0.012 m³) 医疗、 (14, 0.027) 饮用水、 (8, 0.028) 食品、 (6, 0.035) 卫生
因此单服务区装箱可用「按类型计数」的整数规划精确求解，规模极小。

口径（与题面附录 2、S21—S29 一致）：
- 架次 O01->Si->O01，仅服务一个服务区，不得跨区组批
- 不考虑实体无人机与共享电池调度
- 约束：不可拆箱、单箱只装一次、载质量 <= min(机型最大载货, 能量余量反解上限)、
        装载体积 <= 机型可用装载体积、架次能耗 <= (1-rho)*E_use
- 目标：全交付下权衡架次数、总能耗、累计作业时间
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from functools import lru_cache

from core import (AircraftType, Segment, build_segment, load_aircraft_types,
                  load_boxes, load_dem, load_nodes, segment_energy_kwh,
                  Terrain)

TYPE_ORDER = ("MED", "WAT", "FOD", "HYG")
KIND_TO_TYPE = {"医疗物资": "MED", "饮用水": "WAT", "应急食品": "FOD", "生活卫生用品": "HYG"}
# (物资类型, (单箱质量 kg, 单箱体积 m^3))，取自逐箱货箱清单
BOX_SPEC = [
    ("医疗物资", (3.0, 0.012)),
    ("饮用水", (14.0, 0.027)),
    ("应急食品", (8.0, 0.028)),
    ("生活卫生用品", (6.0, 0.035)),
]


@dataclass(frozen=True)
class BoxType:
    key: str
    kind: str
    mass_kg: float
    volume_m3: float


def box_types(boxes: list[dict]) -> tuple[list[BoxType], dict[str, int]]:
    """返回该服务区实际出现的箱型与计数（按固定顺序，便于结果可比）。"""
    present = {KIND_TO_TYPE[b["kind"]] for b in boxes}
    types = [BoxType(KIND_TO_TYPE[k], k, m, v) for k, (m, v) in BOX_SPEC if KIND_TO_TYPE[k] in present]
    counts = {t.key: 0 for t in types}
    for b in boxes:
        counts[KIND_TO_TYPE[b["kind"]]] += 1
    return types, counts


def round_trip_energy(ac: AircraftType, outbound: Segment, inbound: Segment, mass: float) -> float:
    """单点往返架次能耗：去程携带全部载荷，投送后返程空载。"""
    return segment_energy_kwh(ac, mass, outbound) + segment_energy_kwh(ac, 0.0, inbound)


def round_trip_time(ac: AircraftType, outbound: Segment, inbound: Segment, n_boxes: int) -> float:
    """单点往返架次作业时间：准备 + 往返飞行 + 装载 + 交接。"""
    flight = outbound.flight_time_s(ac) + inbound.flight_time_s(ac)
    handling = (n_boxes * ac.load_time_per_box_s
                + ac.handover_base_s
                + n_boxes * ac.handover_per_box_s)
    return ac.setup_time_s + flight + handling


def effective_payload_cap(ac: AircraftType, outbound: Segment, inbound: Segment) -> float:
    limit = (1.0 - ac.reserve_ratio) * ac.battery_kwh
    lo, hi = 0.0, ac.max_payload_kg
    if round_trip_energy(ac, outbound, inbound, hi) <= limit:
        return hi
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if round_trip_energy(ac, outbound, inbound, mid) <= limit:
            lo = mid
        else:
            hi = mid
    return lo


def load_plan(ac: AircraftType, outbound: Segment, inbound: Segment, mass_cap: float,
              counts: dict[str, int], types: list[BoxType], objective: str = "batches",
              reserve_ratio: float | None = None):
    """精确求解单服务区装箱。

    objective:
      - "batches": 最小架次数（并列时最小化能耗）
      - "energy":  给定架次数下最小化能耗（先由 batches 得到最小架次数）
    返回 (架次数, 每架次的类型计数列表, 该方案的能耗)
    """
    keys = [t.key for t in types]
    mass = {t.key: t.mass_kg for t in types}
    vol = {t.key: t.volume_m3 for t in types}
    rho = ac.reserve_ratio if reserve_ratio is None else reserve_ratio
    energy_limit = (1.0 - rho) * ac.battery_kwh

    # 枚举所有可行架次装载（类型计数组合），规模 = prod(count+1)，本题 <= 16*9*9*5 级别
    feasible: list[tuple[tuple[int, ...], float, float, int]] = []
    ranges = [range(counts[k] + 1) for k in keys]
    import itertools
    for combo in itertools.product(*ranges):
        if sum(combo) == 0:
            continue
        m = sum(combo[i] * mass[keys[i]] for i in range(len(keys)))
        v = sum(combo[i] * vol[keys[i]] for i in range(len(keys)))
        if m > mass_cap + 1e-9 or v > ac.volume_m3 + 1e-12:
            continue
        e = round_trip_energy(ac, outbound, inbound, m)
        if e > energy_limit + 1e-12:
            continue
        feasible.append((combo, m, e, sum(combo)))
    feasible.sort(key=lambda x: (-x[3], x[2]))

    # 单箱不可行时（返航余量过大导致连一只最轻箱都装不下），该机型对该服务区不可用
    lightest = min(mass.values()) if mass else 0.0
    if not feasible or round_trip_energy(ac, outbound, inbound, lightest) > energy_limit + 1e-12:
        return None, None, None

    target = tuple(counts[k] for k in keys)

    @lru_cache(maxsize=None)
    def solve(remaining: tuple[int, ...]):
        if all(r == 0 for r in remaining):
            return 0, 0.0, []
        best_n = 10 ** 9
        best_e = float("inf")
        best_plan = None
        for combo, m, e, nb in feasible:
            if all(combo[i] <= remaining[i] for i in range(len(keys))):
                next_remaining = tuple(remaining[i] - combo[i] for i in range(len(keys)))
                sub_n, sub_e, sub_plan = solve(next_remaining)
                cand_n = 1 + sub_n
                cand_e = e + sub_e
                if cand_n < best_n or (cand_n == best_n and cand_e < best_e - 1e-12):
                    best_n = cand_n
                    best_e = cand_e
                    best_plan = [combo] + sub_plan
        if best_plan is None:
            return 10 ** 9, float("inf"), None
        return best_n, best_e, best_plan

    n, energy, plan = solve(target)
    return n, plan, energy


def mixed_load_plan(aircraft: dict[str, AircraftType], outbound: Segment, inbound: Segment,
                    counts: dict[str, int], types: list[BoxType], reserve_ratio: float = 0.20):
    """跨机型精确组批，按架次数、总能耗、总作业时间依次优化。"""
    keys = [t.key for t in types]
    mass = {t.key: t.mass_kg for t in types}
    vol = {t.key: t.volume_m3 for t in types}
    ranges = [range(counts[k] + 1) for k in keys]

    # option = (机型, 箱型计数组合, 质量, 体积, 能耗, 作业时间)
    options = []
    import itertools
    for g, ac in aircraft.items():
        energy_limit = (1.0 - reserve_ratio) * ac.battery_kwh
        for combo in itertools.product(*ranges):
            n_boxes = sum(combo)
            if n_boxes == 0:
                continue
            m = sum(combo[i] * mass[keys[i]] for i in range(len(keys)))
            v = sum(combo[i] * vol[keys[i]] for i in range(len(keys)))
            if m > ac.max_payload_kg + 1e-9 or v > ac.volume_m3 + 1e-12:
                continue
            e = round_trip_energy(ac, outbound, inbound, m)
            if e > energy_limit + 1e-12:
                continue
            t = round_trip_time(ac, outbound, inbound, n_boxes)
            options.append((g, combo, m, v, e, t))

    if not options:
        return None, None, None, None
    options.sort(key=lambda x: (-sum(x[1]), x[4], x[5], x[0]))
    target = tuple(counts[k] for k in keys)

    @lru_cache(maxsize=None)
    def solve(remaining: tuple[int, ...]):
        if all(r == 0 for r in remaining):
            return 0, 0.0, 0.0, []
        best = None
        for option in options:
            combo = option[1]
            if not all(combo[i] <= remaining[i] for i in range(len(keys))):
                continue
            next_remaining = tuple(remaining[i] - combo[i] for i in range(len(keys)))
            sub_n, sub_e, sub_t, sub_plan = solve(next_remaining)
            if sub_plan is None:
                continue
            candidate = (1 + sub_n, option[4] + sub_e, option[5] + sub_t,
                         [option] + sub_plan)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
        return best if best is not None else (10 ** 9, float("inf"), float("inf"), None)

    n, energy, total_time, plan = solve(target)
    return n, plan, energy, total_time


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    acs = load_aircraft_types()
    nodes = load_nodes()
    boxes = load_boxes()
    dem, lat, lon = load_dem()
    terrain = Terrain(dem, lat, lon)

    by_service: dict[str, list[dict]] = {}
    for b in boxes:
        by_service.setdefault(b["service"], []).append(b)

    segs: dict[str, tuple[Segment, Segment]] = {}
    for s in by_service:
        segs[s] = (build_segment(terrain, nodes["O01"], nodes[s]),
                   build_segment(terrain, nodes[s], nodes["O01"]))

    print("== 最大安全载荷 (kg) ==")
    print("  %-6s %8s %8s %8s" % ("服务区", "A", "B", "C"))
    caps: dict[tuple[str, str], float] = {}
    for s in sorted(by_service):
        o, i = segs[s]
        row = []
        for g in "ABC":
            q = effective_payload_cap(acs[g], o, i)
            caps[(s, g)] = q
            row.append(q)
        print("  %-6s %8.2f %8.2f %8.2f" % (s, row[0], row[1], row[2]))

    print("\n== 单服务区最小架次与能耗（按机型独立评价）==")
    print("  %-6s %-14s %6s %10s %10s" % ("服务区", "机型", "架次", "总能耗kWh", "总作业时间s"))
    summary = {}
    for s in sorted(by_service):
        o, i = segs[s]
        types, counts = box_types(by_service[s])
        for g in "ABC":
            ac = acs[g]
            n, plan, energy = load_plan(ac, o, i, caps[(s, g)], counts, types)
            total_time = sum(round_trip_time(ac, o, i, sum(combo)) for combo in plan)
            summary[(s, g)] = (n, energy, total_time, plan)
            print("  %-6s %-14s %6d %10.4f %10.1f" % (s, g, n, energy, total_time))


if __name__ == "__main__":
    main()
