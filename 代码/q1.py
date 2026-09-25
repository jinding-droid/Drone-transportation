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

from core import (AircraftType, Segment, build_segment, charge_time_s,
                  equivalent_range_m, load_aircraft_types, load_boxes, load_dem,
                  load_nodes, segment_energy_kwh, horizontal_energy_kwh,
                  climb_energy_kwh, Terrain)

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
    return segment_energy_kwh(ac, mass, outbound) + segment_energy_kwh(ac, mass, inbound)


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
              counts: dict[str, int], types: list[BoxType], objective: str = "batches"):
    """精确求解单服务区装箱。

    objective:
      - "batches": 最小架次数（并列时最小化能耗）
      - "energy":  给定架次数下最小化能耗（先由 batches 得到最小架次数）
    返回 (架次数, 每架次的类型计数列表, 该方案的能耗, 装载质量)
    """
    keys = [t.key for t in types]
    mass = {t.key: t.mass_kg for t in types}
    vol = {t.key: t.volume_m3 for t in types}
    total = sum(counts.values())

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
        if e > (1.0 - ac.reserve_ratio) * ac.battery_kwh + 1e-12:
            continue
        feasible.append((combo, m, e, sum(combo)))
    feasible.sort(key=lambda x: (-x[3], x[2]))

    # 单箱不可行时（返航余量过大导致连一只最轻箱都装不下），该机型对该服务区不可用
    lightest = min(mass.values()) if mass else 0.0
    if not feasible or round_trip_energy(ac, outbound, inbound, lightest) > (1.0 - ac.reserve_ratio) * ac.battery_kwh + 1e-12:
        return None, None, None

    best: dict[str, object] = {"n": 10 ** 9, "energy": float("inf"), "plan": None}

    target = tuple(counts[k] for k in keys)

    def dfs(remaining: tuple[int, ...], plan: list[tuple[int, ...]], e_acc: float) -> None:
        if len(plan) >= best["n"]:
            return
        if all(r == 0 for r in remaining):
            if len(plan) < best["n"] or (len(plan) == best["n"] and e_acc < best["energy"] - 1e-12):
                best.update(n=len(plan), energy=e_acc, plan=list(plan))
            return
        # 下界剪枝
        max_boxes = max(f[3] for f in feasible)
        lb = math.ceil(sum(remaining) / max_boxes)
        if len(plan) + lb > best["n"]:
            return
        for combo, m, e, nb in feasible:
            if all(combo[i] <= remaining[i] for i in range(len(keys))):
                if all(combo[i] == 0 for i in range(len(keys))):
                    continue
                dfs(tuple(remaining[i] - combo[i] for i in range(len(keys))),
                    plan + [combo], e_acc + e)

    dfs(target, [], 0.0)
    return best["n"], best["plan"], best["energy"]


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
            # 作业时间：准备 + 往返飞行 + 装载 + 交接
            per_flight = 0.0
            for seg in (o, i):
                per_flight += (seg.climb_m / ac.climb_speed_ms + seg.horizontal_m / ac.cruise_speed_ms
                               + seg.descent_m / ac.descent_speed_ms)
            total_time = n * (ac.setup_time_s + per_flight)
            summary[(s, g)] = (n, energy, total_time, plan)
            print("  %-6s %-14s %6d %10.4f %10.1f" % (s, g, n, energy, total_time))


if __name__ == "__main__":
    main()
