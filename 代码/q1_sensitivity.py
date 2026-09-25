# -*- coding: utf-8 -*-
"""问题一扩展：返航安全余量灵敏度 + 指标权衡 + 结果导出。

导出内容：
  结果/Q1_最大安全载荷.csv
  结果/Q1_组批方案.csv            （推荐机型的逐服务区组批明细）
  结果/Q1_机型对比.csv            （每服务区每机型的架次/能耗/时间）
  结果/Q1_安全余量灵敏度.csv      （rho 扫描下最大安全载荷变化）
"""

from __future__ import annotations

import csv
import os
import sys

from core import (build_segment, load_aircraft_types, load_boxes, load_dem,
                  load_nodes, Terrain)
from q1 import box_types, effective_payload_cap, load_plan, round_trip_energy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "结果")
RHO_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def flight_time(ac, o, i) -> float:
    return sum(seg.climb_m / ac.climb_speed_ms + seg.horizontal_m / ac.cruise_speed_ms
               + seg.descent_m / ac.descent_speed_ms for seg in (o, i))


def cap_with_rho(ac, o, i, rho: float) -> float:
    limit = (1.0 - rho) * ac.battery_kwh

    def used(q):
        return round_trip_energy(ac, o, i, q)

    lo, hi = 0.0, ac.max_payload_kg
    if used(hi) <= limit:
        return hi
    for _ in range(100):
        mid = (lo + hi) / 2
        if used(mid) <= limit:
            lo = mid
        else:
            hi = mid
    return lo


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(OUT, exist_ok=True)
    acs = load_aircraft_types()
    nodes = load_nodes()
    boxes = load_boxes()
    dem, lat, lon = load_dem()
    terrain = Terrain(dem, lat, lon)
    by_service: dict[str, list[dict]] = {}
    for b in boxes:
        by_service.setdefault(b["service"], []).append(b)
    segs = {s: (build_segment(terrain, nodes["O01"], nodes[s]),
                build_segment(terrain, nodes[s], nodes["O01"])) for s in by_service}

    # 1) 最大安全载荷
    with open(os.path.join(OUT, "Q1_最大安全载荷.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["服务区编号", "A最大安全载荷kg", "B最大安全载荷kg", "C最大安全载荷kg",
                    "A是否满载", "B是否满载", "C是否满载"])
        for s in sorted(by_service):
            o, i = segs[s]
            row = []
            flags = []
            for g in "ABC":
                ac = acs[g]
                q = effective_payload_cap(ac, o, i)
                row.append(round(q, 3))
                flags.append("是" if abs(q - ac.max_payload_kg) < 1e-6 else "否")
            w.writerow([s] + row + flags)

    # 2) 机型对比与推荐
    rows = []
    for s in sorted(by_service):
        o, i = segs[s]
        types, counts = box_types(by_service[s])
        for g in "ABC":
            ac = acs[g]
            n, plan, energy = load_plan(ac, o, i, effective_payload_cap(ac, o, i), counts, types)
            t = n * (ac.setup_time_s + flight_time(ac, o, i))
            rows.append(dict(service=s, ac=g, batches=n, energy=energy, time=t, plan=plan,
                             types=types, counts=counts))

    with open(os.path.join(OUT, "Q1_机型对比.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["服务区编号", "机型", "架次数", "总能耗kWh", "总作业时间s"])
        for r in rows:
            w.writerow([r["service"], r["ac"], r["batches"],
                        round(r["energy"], 4), round(r["time"], 1)])

    # 3) 推荐：三目标（架次、能耗、时间）折中 —— 用归一化加权，权重 1/3 各
    print("== 各机型全服务区汇总 ==")
    print("  %-4s %8s %12s %12s" % ("机型", "总架次", "总能耗kWh", "总时间s"))
    agg = {}
    for g in "ABC":
        sub = [r for r in rows if r["ac"] == g]
        agg[g] = (sum(r["batches"] for r in sub), sum(r["energy"] for r in sub),
                  sum(r["time"] for r in sub))
        print("  %-4s %8d %12.3f %12.1f" % (g, *agg[g]))

    # 分机型独立汇总（每题只选一个机型时的口径）
    print("\n== 分服务区最优机型（按归一化综合指标）==")
    best = {}
    for s in sorted(by_service):
        sub = [r for r in rows if r["service"] == s]
        nb = [r["batches"] for r in sub]
        ne = [r["energy"] for r in sub]
        nt = [r["time"] for r in sub]
        def norm(xs, x):
            lo, hi = min(xs), max(xs)
            return 0.0 if hi == lo else (x - lo) / (hi - lo)
        score = {r["ac"]: norm(nb, r["batches"]) + norm(ne, r["energy"]) + norm(nt, r["time"]) for r in sub}
        pick = min(score, key=score.get)
        best[s] = pick
        print("  %-6s -> %s   (评分 %s)" % (s, pick,
              ", ".join("%s=%.3f" % (k, score[k]) for k in "ABC")))

    # 4) 组批方案导出（推荐机型）
    with open(os.path.join(OUT, "Q1_组批方案.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["架次编号", "服务区编号", "机型编号", "货箱组合", "总质量kg", "总体积m³",
                    "往返时间s", "架次能耗kWh", "返航SOC%"])
        k = 0
        for s in sorted(by_service):
            r = next(x for x in rows if x["service"] == s and x["ac"] == best[s])
            ac = acs[best[s]]
            ft = flight_time(ac, *segs[s])
            for combo in r["plan"]:
                k += 1
                m = sum(combo[j] * r["types"][j].mass_kg for j in range(len(r["types"])))
                v = sum(combo[j] * r["types"][j].volume_m3 for j in range(len(r["types"])))
                e = round_trip_energy(ac, *segs[s], m)
                soc = max(0.0, 1.0 - e / ac.battery_kwh)
                desc = " + ".join("%s×%d" % (r["types"][j].kind, combo[j])
                                  for j in range(len(r["types"])) if combo[j] > 0)
                w.writerow(["B%03d" % k, s, best[s], desc, round(m, 2), round(v, 4),
                            round(ac.setup_time_s + ft, 1), round(e, 4), round(soc * 100, 2)])

    # 5) 安全余量灵敏度
    print("\n== 返航安全余量灵敏度：最大安全载荷 (kg) ==")
    hdr = "  %-6s" % "rho"
    for s in ("S001", "S004", "S008", "S015"):
        hdr += " %14s" % s
    print(hdr)
    with open(os.path.join(OUT, "Q1_安全余量灵敏度.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["返航安全余量rho", "服务区编号", "机型", "最大安全载荷kg", "架次数", "总能耗kWh"])
        for rho in RHO_GRID:
            line = "  %-6.2f" % rho
            for s in ("S001", "S004", "S008", "S015"):
                o, i = segs[s]
                vals = []
                for g in "ABC":
                    ac = acs[g]
                    q = cap_with_rho(ac, o, i, rho)
                    types, counts = box_types(by_service[s])
                    n, plan, e = load_plan(ac, o, i, q, counts, types)
                    w.writerow([rho, s, g, round(q, 3),
                                "-" if n is None else n,
                                "-" if e is None else round(e, 4)])
                    vals.append(q)
                line += " %14s" % (" ".join("%.1f" % v for v in vals))
            print(line)

    print("\n导出完成：", OUT)


if __name__ == "__main__":
    main()
