# -*- coding: utf-8 -*-
"""D 题公共物理内核：数据读取、航段提取、时间与能耗计算。

计算口径全部来自题面附录 2 与附件参数，不做任何自创简化。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import openpyxl
import scipy.io as sio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "数据", "无人机应急物资运输基础数据")
GEO_DIR = os.path.join(ROOT, "数据", "镇龙乡地理空间数据", "镇龙乡及周边地理数据")
DEM_MAT = os.path.join(GEO_DIR, "数字高程模型数据（DEM）", "镇龙乡及周边30米DEM.mat")

# 附录 2：巡航海拔 = 航段所经 DEM 像元最高地面高程 + 50 m
CRUISE_CLEARANCE_M = 50.0
# 附录 2：服务区作业高度 = 地面海拔 + 30 m
SERVICE_OP_ALT_AGL = 30.0
# 通信链路参数：G01 天线离地高度
GATEWAY_ANTENNA_AGL = 20.0
# 地球平均半径（经纬度 → 水平距离）
EARTH_R_M = 6371008.8
NO_DATA = -32767.0


@dataclass(frozen=True)
class AircraftType:
    code: str
    name: str
    empty_mass_kg: float
    max_payload_kg: float
    volume_m3: float
    cruise_speed_ms: float
    range_empty_m: float
    range_full_m: float
    battery_kwh: float
    reserve_ratio: float
    setup_time_s: float
    load_time_per_box_s: float
    handover_base_s: float
    handover_per_box_s: float
    climb_speed_ms: float
    descent_speed_ms: float
    climb_eff: float
    descent_eff: float


@dataclass(frozen=True)
class Node:
    code: str
    name: str
    lon: float
    lat: float
    elevation_m: float
    population: int = 0

    @property
    def op_alt_m(self) -> float:
        """作业高度（海拔，m）。O01 取地面海拔，服务区取地面海拔 + 30 m。"""
        if self.code == "O01":
            return self.elevation_m
        return self.elevation_m + SERVICE_OP_ALT_AGL


@dataclass(frozen=True)
class Segment:
    """一条水平直线航段的几何与时间属性（与载荷无关）。"""

    src: str
    dst: str
    horizontal_m: float
    cruise_alt_m: float
    ridge_alt_m: float
    climb_m: float
    descent_m: float

    def flight_time_s(self, ac: AircraftType) -> float:
        return (
            self.climb_m / ac.climb_speed_ms
            + self.horizontal_m / ac.cruise_speed_ms
            + self.descent_m / ac.descent_speed_ms
        )


def load_aircraft_types() -> dict[str, AircraftType]:
    wb = openpyxl.load_workbook(os.path.join(DATA_DIR, "运输无人机数据.xlsx"), data_only=True)
    ws = wb["数据"]
    rows = list(ws.iter_rows(values_only=True))
    header_idx = next(i for i, r in enumerate(rows) if r and r[0] == "机型编号")
    out: dict[str, AircraftType] = {}
    for r in rows[header_idx + 1:]:
        if not r or r[0] is None:
            break
        if r[0] not in ("A", "B", "C"):
            continue
        out[r[0]] = AircraftType(
            code=r[0], name=r[1], empty_mass_kg=float(r[2]), max_payload_kg=float(r[3]),
            volume_m3=float(r[4]), cruise_speed_ms=float(r[5]), range_empty_m=float(r[6]),
            range_full_m=float(r[7]), battery_kwh=float(r[8]), reserve_ratio=float(r[9]) / 100.0,
            setup_time_s=float(r[10]), load_time_per_box_s=float(r[11]),
            handover_base_s=float(r[12]), handover_per_box_s=float(r[13]),
            climb_speed_ms=float(r[14]), descent_speed_ms=float(r[15]),
            climb_eff=float(r[16]), descent_eff=float(r[17]),
        )
    return out


def load_relay_type() -> dict:
    wb = openpyxl.load_workbook(os.path.join(DATA_DIR, "中继无人机数据.xlsx"), data_only=True)
    ws = wb["数据"]
    rows = list(ws.iter_rows(values_only=True))
    header_idx = next(i for i, r in enumerate(rows) if r and r[0] == "机型编号")
    r = rows[header_idx + 1]
    keys = ["code", "name", "empty_mass_kg", "comm_module_kg", "takeoff_mass_kg",
            "cruise_speed_ms", "cruise_power_kw", "battery_kwh", "reserve_ratio_pct",
            "setup_time_s", "link_setup_s", "turnaround_s", "climb_speed_ms",
            "descent_speed_ms", "climb_eff", "descent_eff", "hover_power_kw",
            "comm_extra_power_kw", "max_hover_agl_m"]
    return dict(zip(keys, r[:len(keys)]))


def load_relay_fleet() -> tuple[list[str], int, float]:
    """返回（中继无人机编号、能源组件库存、等效完全充电时间）。"""
    wb = openpyxl.load_workbook(os.path.join(DATA_DIR, "中继无人机数据.xlsx"), data_only=True)
    rows = list(wb["数据"].iter_rows(values_only=True))
    start = next(i for i, r in enumerate(rows) if r and r[0] == "中继无人机编号")
    drones: list[str] = []
    for r in rows[start + 1:]:
        if not r or not r[0] or not str(r[0]).startswith("R"):
            break
        drones.append(str(r[0]))
    start = next(i for i, r in enumerate(rows) if r and r[0] == "机型编号" and i > start)
    row = rows[start + 1]
    return drones, int(row[1]), float(row[2])


def load_fleet() -> tuple[list[tuple[str, str]], dict[str, int], dict[str, float]]:
    """返回 (逐架无人机清单, 共享电池库存, 等效完全充电时间)。"""
    wb = openpyxl.load_workbook(os.path.join(DATA_DIR, "运输无人机数据.xlsx"), data_only=True)
    rows = list(wb["数据"].iter_rows(values_only=True))
    start = next(i for i, r in enumerate(rows) if r and r[0] == "无人机编号")
    drones = []
    for r in rows[start + 1:]:
        if not r or not r[0] or not str(r[0]).startswith("U"):
            break
        drones.append((str(r[0]), str(r[1])))
    start = next(i for i, r in enumerate(rows) if r and r[0] == "机型编号" and i > start)
    stock, t_full = {}, {}
    for r in rows[start + 1:]:
        if not r or r[0] is None:
            break
        if r[0] not in ("A", "B", "C"):
            continue
        stock[r[0]] = int(r[1])
        t_full[r[0]] = float(r[2])
    return drones, stock, t_full


def load_nodes() -> dict[str, Node]:
    wb = openpyxl.load_workbook(os.path.join(DATA_DIR, "调度中心与服务区.xlsx"), data_only=True)
    rows = list(wb["数据"].iter_rows(values_only=True))
    idx = next(i for i, r in enumerate(rows) if r and r[0] == "调度中心编号")
    o = rows[idx + 1]
    nodes = {"O01": Node("O01", str(o[1]), float(o[2]), float(o[3]), float(o[4]), 0)}
    idx = next(i for i, r in enumerate(rows) if r and r[0] == "服务区编号")
    for r in rows[idx + 1:]:
        if not r or not r[0]:
            continue
        nodes[str(r[0])] = Node(str(r[0]), str(r[1]), float(r[2]), float(r[3]),
                               float(r[4]), int(r[5]) if r[5] is not None else 0)
    return nodes


def load_boxes() -> list[dict]:
    wb = openpyxl.load_workbook(os.path.join(DATA_DIR, "物资需求与配送时限.xlsx"), data_only=True)
    rows = list(wb["逐箱货箱清单"].iter_rows(values_only=True))
    header = rows[0]
    assert header[0] == "货箱编号", header
    boxes = []
    for r in rows[1:]:
        if not r or not r[0]:
            continue
        boxes.append({
            "id": str(r[0]), "service": str(r[1]), "kind": str(r[2]),
            "mass_kg": float(r[3]), "volume_m3": float(r[4]),
            "is_first_batch": (r[5] == "是"),
            "first_batch_deadline_s": float(r[6]) if r[6] is not None else None,
            "expected_time_s": float(r[7]) if r[7] is not None else None,
            "priority": float(r[8]) if r[8] is not None else None,
        })
    return boxes


def load_link_params() -> dict:
    wb = openpyxl.load_workbook(os.path.join(DATA_DIR, "通信链路参数.xlsx"), data_only=True)
    rows = list(wb["数据"].iter_rows(values_only=True))
    out: dict[str, float] = {}
    semantic = {
        ("传播参数", "f"): "frequency_mhz",
        ("传播参数", "Lsys"): "system_loss_db",
        ("传播参数", "Lobs"): "obstacle_loss_db",
        ("接收参数", "Psens"): "sensitivity_dbm",
        ("接收参数", "M"): "fade_margin_db",
        ("运输无人机", "Pt"): "transport_pt_dbm",
        ("运输无人机", "G"): "transport_gain_dbi",
        ("中继接入端", "Pt"): "relay_access_pt_dbm",
        ("中继接入端", "G"): "relay_access_gain_dbi",
        ("中继回传端", "Pt"): "relay_backhaul_pt_dbm",
        ("中继回传端", "G"): "relay_backhaul_gain_dbi",
        ("固定网关 G01", "Pt"): "gateway_pt_dbm",
        ("固定网关 G01", "G"): "gateway_gain_dbi",
        ("固定网关 G01", "hG"): "gateway_agl_m",
    }
    for r in rows:
        if (not r or len(r) < 5 or r[0] is None or r[3] is None
                or not isinstance(r[4], (int, float))):
            continue
        key = semantic.get((str(r[0]), str(r[3])))
        if key is not None:
            out[key] = float(r[4])
    missing = set(semantic.values()) - set(out)
    if missing:
        raise ValueError(f"通信链路参数缺失：{sorted(missing)}")
    return out


def load_dem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = sio.loadmat(DEM_MAT)
    dem = np.asarray(m["dem"], dtype=np.float64)
    lat = np.asarray(m["latitude"], dtype=np.float64).ravel()
    lon = np.asarray(m["longitude"], dtype=np.float64).ravel()
    return dem, lat, lon


class Terrain:
    """DEM 的高程查询与航段地形净空计算。"""

    def __init__(self, dem: np.ndarray, lat: np.ndarray, lon: np.ndarray):
        self.dem = dem
        self.lat = lat
        self.lon = lon
        self._valid = dem > (NO_DATA + 1.0)
        self.dlat = float(lat[1] - lat[0])
        self.dlon = float(lon[1] - lon[0])

    def elevation(self, lon: float, lat: float) -> float:
        i = int(round((lat - self.lat[0]) / self.dlat))
        j = int(round((lon - self.lon[0]) / self.dlon))
        i = min(max(i, 0), self.dem.shape[0] - 1)
        j = min(max(j, 0), self.dem.shape[1] - 1)
        v = self.dem[i, j]
        return float(v) if v > NO_DATA + 1.0 else float(np.nanmax(self.dem))

    def ridge_elevation(self, lon1: float, lat1: float, lon2: float, lat2: float,
                        samples: int | None = None) -> float:
        """航段沿线 DEM 像元最高地面高程（含两端点）。"""
        if samples is None:
            di = abs(lat2 - lat1) / abs(self.dlat)
            dj = abs(lon2 - lon1) / abs(self.dlon)
            # 密采样避免斜向航段漏掉仅从边缘穿过的像元。
            samples = max(2, int(math.ceil(max(di, dj) * 8.0)) + 2)
        lons = np.linspace(lon1, lon2, samples)
        lats = np.linspace(lat1, lat2, samples)
        ii = np.round((lats - self.lat[0]) / self.dlat).astype(int)
        jj = np.round((lons - self.lon[0]) / self.dlon).astype(int)
        ii = np.clip(ii, 0, self.dem.shape[0] - 1)
        jj = np.clip(jj, 0, self.dem.shape[1] - 1)
        vals = self.dem[ii, jj]
        vals = vals[vals > NO_DATA + 1.0]
        if vals.size == 0:
            return float(np.nanmax(self.dem))
        return float(vals.max())

    def points_along(self, lon1: float, lat1: float, lon2: float, lat2: float,
                     samples: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回航段沿线采样点的 (lon, lat, 地面高程)。"""
        if samples is None:
            di = abs(lat2 - lat1) / abs(self.dlat)
            dj = abs(lon2 - lon1) / abs(self.dlon)
            samples = max(2, int(math.ceil(max(di, dj))) + 1)
        lons = np.linspace(lon1, lon2, samples)
        lats = np.linspace(lat1, lat2, samples)
        ii = np.clip(np.round((lats - self.lat[0]) / self.dlat).astype(int), 0, self.dem.shape[0] - 1)
        jj = np.clip(np.round((lons - self.lon[0]) / self.dlon).astype(int), 0, self.dem.shape[1] - 1)
        vals = self.dem[ii, jj]
        return lons, lats, vals


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R_M * math.asin(math.sqrt(a))


def build_segment(terrain: Terrain, src: Node, dst: Node) -> Segment:
    horizontal = haversine_m(src.lon, src.lat, dst.lon, dst.lat)
    ridge = terrain.ridge_elevation(src.lon, src.lat, dst.lon, dst.lat)
    cruise_alt = ridge + CRUISE_CLEARANCE_M
    climb = max(0.0, cruise_alt - src.op_alt_m)
    descent = max(0.0, cruise_alt - dst.op_alt_m)
    return Segment(src.code, dst.code, horizontal, cruise_alt, ridge, climb, descent)


def build_all_segments(terrain: Terrain, nodes: dict[str, Node]) -> dict[tuple[str, str], Segment]:
    codes = list(nodes)
    segs: dict[tuple[str, str], Segment] = {}
    for a in codes:
        for b in codes:
            if a == b:
                continue
            segs[(a, b)] = build_segment(terrain, nodes[a], nodes[b])
    return segs


def equivalent_range_m(ac: AircraftType, payload_kg: float) -> float:
    """附录 2：L_g(q) = L_g0 - (L_g0 - L_gF) * (q / Q_g)^{3/2}。"""
    q = min(max(payload_kg, 0.0), ac.max_payload_kg)
    ratio = q / ac.max_payload_kg
    return ac.range_empty_m - (ac.range_empty_m - ac.range_full_m) * ratio ** 1.5


def unit_energy_kwh_per_m(ac: AircraftType, payload_kg: float) -> float:
    """水平巡航单位距离能耗：单组电池可用能量恰好覆盖等效航程 L_g(q)。"""
    return ac.battery_kwh / equivalent_range_m(ac, payload_kg)


def horizontal_energy_kwh(ac: AircraftType, payload_kg: float, horizontal_m: float) -> float:
    return unit_energy_kwh_per_m(ac, payload_kg) * horizontal_m


def climb_energy_kwh(ac: AircraftType, payload_kg: float, climb_m: float) -> float:
    """爬升附加能耗：（空机质量 + 载荷）重力势能 / 爬升能耗效率。下降效率取 0，不单独计。"""
    mass = ac.empty_mass_kg + payload_kg
    return mass * 9.80665 * climb_m / ac.climb_eff / 3.6e6


def segment_energy_kwh(ac: AircraftType, payload_kg: float, seg: Segment) -> float:
    return (horizontal_energy_kwh(ac, payload_kg, seg.horizontal_m)
            + climb_energy_kwh(ac, payload_kg, seg.climb_m))


def max_safe_payload_kg(ac: AircraftType, seg: Segment, tol: float = 1e-9) -> float:
    """二分反解最大安全载荷：满足 E(q) <= (1 - rho) * E_use 的最大 q（不超过 Q_g）。"""
    limit = (1.0 - ac.reserve_ratio) * ac.battery_kwh

    def used(q: float) -> float:
        return segment_energy_kwh(ac, q, seg)

    lo, hi = 0.0, ac.max_payload_kg
    if used(hi) <= limit:
        return hi
    if used(lo) > limit:
        return 0.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if used(mid) <= limit:
            lo = mid
        else:
            hi = mid
    return lo


def charge_time_s(t_full_s: float, soc: float) -> float:
    """附录 2：两阶段等效充电模型。"""
    s = min(max(soc, 0.0), 1.0)
    if s < 0.90:
        return t_full_s * (0.65 * (0.90 - s) / 0.90 + 0.35)
    return t_full_s * 0.35 * (1.0 - s) / 0.10


def mission_time_s(ac: AircraftType, segments: list[Segment], n_boxes_at_stop: list[int]) -> float:
    """架次总时间 = 固定准备 + 各航段时间 + 各服务区装载/交接时间。"""
    t = ac.setup_time_s
    for i, seg in enumerate(segments):
        t += seg.flight_time_s(ac)
        if seg.dst != "O01":
            n = n_boxes_at_stop[i]
            t += ac.handover_base_s + ac.handover_per_box_s * n
    return t


def load_time_s(ac: AircraftType, n_boxes: int) -> float:
    return ac.load_time_per_box_s * n
