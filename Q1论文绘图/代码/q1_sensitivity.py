# -*- coding: utf-8 -*-
"""问题一扩展：返航安全余量灵敏度 + 指标权衡 + 结果导出。

导出内容：
  结果/Q1_最大安全载荷.csv
  结果/Q1_组批方案.csv            （推荐机型的逐服务区组批明细）
  结果/Q1_机型对比.csv            （每服务区每机型的架次/能耗/时间）
  结果/Q1_安全余量灵敏度.csv      （rho 扫描下最大安全载荷变化）
  结果/Q1_全局方案灵敏度.csv      （rho 扫描下联合最优方案变化）
"""

from __future__ import annotations

import csv
import os
import sys

from core import (build_segment, load_aircraft_types, load_boxes, load_dem,
                  load_nodes, Terrain)
from q1 import (KIND_TO_TYPE, box_types, effective_payload_cap, load_plan,
                mixed_load_plan, round_trip_energy, round_trip_time)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "结果")
RHO_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


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


def plan_time(ac, o, i, plan) -> float:
    return sum(round_trip_time(ac, o, i, sum(combo)) for combo in plan)


def assign_box_ids(service_boxes: list[dict], types, combo: tuple[int, ...]) -> list[str]:
    """按箱型计数组合取出真实货箱编号；会原地消耗 service_boxes 中的已分配箱。"""
    selected: list[str] = []
    for j, count in enumerate(combo):
        if count <= 0:
            continue
        key = types[j].key
        taken = 0
        keep = []
        for box in service_boxes:
            if taken < count and KIND_TO_TYPE[box["kind"]] == key:
                selected.append(box["id"])
                taken += 1
            else:
                keep.append(box)
        service_boxes[:] = keep
        if taken != count:
            raise RuntimeError(f"服务区货箱分配失败：{key} 需要 {count} 个，仅取到 {taken} 个")
    return selected


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
            t = plan_time(ac, o, i, plan)
            rows.append(dict(service=s, ac=g, batches=n, energy=energy, time=t, plan=plan,
                             types=types, counts=counts))

    with open(os.path.join(OUT, "Q1_机型对比.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["服务区编号", "机型", "架次数", "总能耗kWh", "总作业时间s"])
        for r in rows:
            w.writerow([r["service"], r["ac"], r["batches"],
                        round(r["energy"], 4), round(r["time"], 1)])

    # 3) 推荐：跨机型联合精确优化，依次最小化架次数、能耗与作业时间。
    print("== 各机型全服务区汇总 ==")
    print("  %-4s %8s %12s %12s" % ("机型", "总架次", "总能耗kWh", "总时间s"))
    agg = {}
    for g in "ABC":
        sub = [r for r in rows if r["ac"] == g]
        agg[g] = (sum(r["batches"] for r in sub), sum(r["energy"] for r in sub),
                  sum(r["time"] for r in sub))
        print("  %-4s %8d %12.3f %12.1f" % (g, *agg[g]))

    print("\n== 分服务区跨机型联合最优方案 ==")
    mixed = {}
    for s in sorted(by_service):
        types, counts = box_types(by_service[s])
        n, plan, energy, total_time = mixed_load_plan(acs, *segs[s], counts, types)
        mixed[s] = dict(batches=n, plan=plan, energy=energy, time=total_time, types=types)
        models = "+".join(option[0] for option in plan)
        print("  %-6s -> %-5s  架次=%d  能耗=%.4f kWh  时间=%.1f s"
              % (s, models, n, energy, total_time))

    # 4) 组批方案导出（跨机型联合最优方案）
    with open(os.path.join(OUT, "Q1_组批方案.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["架次编号", "服务区编号", "机型编号", "货箱编号列表", "总质量kg", "总体积m³",
                    "往返时间s", "架次能耗kWh", "返航SOC%"])
        k = 0
        for s in sorted(by_service):
            r = mixed[s]
            remaining_boxes = sorted(by_service[s], key=lambda x: x["id"])
            for g, combo, m, v, e, t in r["plan"]:
                k += 1
                ac = acs[g]
                soc = max(0.0, 1.0 - e / ac.battery_kwh)
                ids = assign_box_ids(remaining_boxes, r["types"], combo)
                w.writerow(["B%03d" % k, s, g, ",".join(ids), round(m, 2), round(v, 4),
                            round(t, 1), round(e, 4), round(soc * 100, 2)])

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
                    n, plan, e = load_plan(ac, o, i, q, counts, types, reserve_ratio=rho)
                    w.writerow([rho, s, g, round(q, 3),
                                "-" if n is None else n,
                                "-" if e is None else round(e, 4)])
                    vals.append(q)
                line += " %14s" % (" ".join("%.1f" % v for v in vals))
            print(line)

    with open(os.path.join(OUT, "Q1_全局方案灵敏度.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["返航安全余量rho", "可行性", "架次数", "总能耗kWh", "累计作业时间s",
                    "A架次", "B架次", "C架次"])
        for rho in RHO_GRID:
            all_options = []
            feasible = True
            for s in sorted(by_service):
                types, counts = box_types(by_service[s])
                n, plan, energy, total_time = mixed_load_plan(
                    acs, *segs[s], counts, types, reserve_ratio=rho)
                if plan is None:
                    feasible = False
                    break
                all_options.extend(plan)
            if not feasible:
                w.writerow([rho, "不可行", "-", "-", "-", "-", "-", "-"])
                continue
            model_counts = {g: sum(option[0] == g for option in all_options) for g in "ABC"}
            w.writerow([
                rho, "可行", len(all_options),
                round(sum(option[4] for option in all_options), 4),
                round(sum(option[5] for option in all_options), 1),
                model_counts["A"], model_counts["B"], model_counts["C"],
            ])

    print("\n导出完成：", OUT)


if __name__ == "__main__":
    main()
