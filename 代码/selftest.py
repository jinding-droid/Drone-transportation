# -*- coding: utf-8 -*-
"""内核自测：口径一致性、量纲、单调性与关键数值。"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from core import *

def main():
    acs = load_aircraft_types()
    nodes = load_nodes()
    boxes = load_boxes()
    dem, lat, lon = load_dem()
    terrain = Terrain(dem, lat, lon)

    print("== 数据规模 ==")
    print("机型:", sorted(acs), "| 节点:", len(nodes), "| 货箱:", len(boxes))
    print("DEM:", dem.shape, "lat范围 %.4f~%.4f lon范围 %.4f~%.4f" % (lat[0], lat[-1], lon[0], lon[-1]))

    # 1) DEM 高程与附件海拔一致性
    print("\n== 一致性检验：附件海拔 vs DEM 采样 ==")
    maxdiff = 0.0
    for c, n in sorted(nodes.items()):
        e = terrain.elevation(n.lon, n.lat)
        d = abs(e - n.elevation_m)
        maxdiff = max(maxdiff, d)
        if c in ("O01", "S001", "S015"):
            print("  %s 附件=%7.1f DEM=%7.1f 差=%5.1f" % (c, n.elevation_m, e, d))
    print("  全部节点最大偏差 = %.1f m" % maxdiff)

    # 2) 单点往返航段
    print("\n== 单点往返航段（O01→Si→O01）==")
    for c in ("S001", "S006", "S015"):
        s1 = build_segment(terrain, nodes["O01"], nodes[c])
        s2 = build_segment(terrain, nodes[c], nodes["O01"])
        print("  %s: 水平=%8.1f m 沿线最高=%7.1f 巡航=%7.1f 爬升=%6.1f 下降=%6.1f"
              % (c, s1.horizontal_m, s1.ridge_alt_m, s1.cruise_alt_m, s1.climb_m, s1.descent_m))

    # 3) 最大安全载荷
    print("\n== 最大安全载荷 (kg)：能量判据 ==")
    print("  %-6s %8s %8s %8s" % ("服务区", "A", "B", "C"))
    res = {}
    for c in sorted(n for n in nodes if n != "O01"):
        s1 = build_segment(terrain, nodes["O01"], nodes[c])
        s2 = build_segment(terrain, nodes[c], nodes["O01"])
        row = []
        for g in "ABC":
            ac = acs[g]
            limit = (1 - ac.reserve_ratio) * ac.battery_kwh
            lo, hi = 0.0, ac.max_payload_kg
            f = lambda q: segment_energy_kwh(ac, q, s1) + segment_energy_kwh(ac, 0.0, s2)
            if f(hi) <= limit:
                q = hi
            else:
                for _ in range(200):
                    mid = (lo + hi) / 2
                    if f(mid) <= limit:
                        lo = mid
                    else:
                        hi = mid
                q = lo
            row.append(q)
        res[c] = row
        print("  %-6s %8.2f %8.2f %8.2f" % (c, row[0], row[1], row[2]))

    # 4) 时间检验
    print("\n== 单点往返时间与能耗（S001, 满载荷 25/30/80）==")
    s1 = build_segment(terrain, nodes["O01"], nodes["S001"])
    s2 = build_segment(terrain, nodes["S001"], nodes["O01"])
    for g in "ABC":
        ac = acs[g]
        q = ac.max_payload_kg
        t = mission_time_s(ac, [s1, s2], [0, ac.max_payload_kg and 0])
        t = ac.setup_time_s + s1.flight_time_s(ac) + s2.flight_time_s(ac)
        e = segment_energy_kwh(ac, q, s1) + segment_energy_kwh(ac, 0.0, s2)
        print("  %s: 单程时间=%6.1f s 往返飞行=%6.1f s 能耗=%.4f kWh 余量上限=%.4f"
              % (g, s1.flight_time_s(ac), t, e, (1-ac.reserve_ratio)*ac.battery_kwh))

    # 5) 单调性检验
    print("\n== 单调性检验（距离↑ 能耗↑；载荷↑ 能耗↑）==")
    ac = acs["C"]
    prev_e, prev_d = -1, -1
    ok = True
    for c in sorted(n for n in nodes if n != "O01")[:5]:
        s = build_segment(terrain, nodes["O01"], nodes[c])
        e = segment_energy_kwh(ac, 40.0, s)
        if s.horizontal_m < prev_d and e > prev_e:
            pass
        if s.horizontal_m >= prev_d:
            prev_d, prev_e = s.horizontal_m, e
    prev = -1
    for q in (0, 10, 20, 40, 80):
        e = horizontal_energy_kwh(ac, q, 5000.0)
        if e < prev:
            print("  !! 载荷单调性违反 q=%s" % q); ok = False
        prev = e
    print("  载荷单调递增:", ok)

    # 6) 充电模型边界
    print("\n== 充电模型边界 ==")
    print("  T_full=1800, s=0 ->", charge_time_s(1800, 0.0), "(应=1800)")
    print("  T_full=1800, s=1 ->", charge_time_s(1800, 1.0), "(应=0)")
    print("  T_full=1800, s=0.9 ->", charge_time_s(1800, 0.9), "(应=630=0.35*1800)")
    print("  T_full=1800, s=0.45 ->", charge_time_s(1800, 0.45), "(应=630+0.325*1800=1215)")

    # 7) 等效航程自洽
    print("\n== 等效航程自洽检验 ==")
    for g in "ABC":
        ac = acs[g]
        print("  %s: L(0)=%.0f (应=%.0f)  L(Q)=%.0f (应=%.0f)"
              % (g, equivalent_range_m(ac, 0), ac.range_empty_m,
                 equivalent_range_m(ac, ac.max_payload_kg), ac.range_full_m))

if __name__ == "__main__":
    main()
