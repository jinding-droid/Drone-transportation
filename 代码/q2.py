# -*- coding: utf-8 -*-
"""问题二：异构运输无人机的多点、多架次事件驱动调度。

求解框架：
1. 随机化顺序插入生成货箱组批；
2. 对每个架次精确枚举服务区访问顺序和可用机型；
3. 按事件时刻分配实体无人机与同机型共享电池；
4. 以硬时限可行性为前提，综合优化及时性、完工时间、能耗和架次数。

所有航段时间与能耗均调用 core.py，与问题一保持同一物理口径。
"""

from __future__ import annotations

import csv
import argparse
import itertools
import math
import os
import random
import sys
from dataclasses import dataclass
from functools import lru_cache

from core import (AircraftType, Segment, Terrain, build_all_segments, charge_time_s,
                  load_aircraft_types, load_boxes, load_dem, load_fleet, load_nodes,
                  segment_energy_kwh)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, "结果")
MAX_STOPS = 5
PROTECTED_WAVE_TYPE = {
    "S001": "C", "S002": "C", "S006": "B", "S007": "B",
    "S010": "A", "S012": "A", "S013": "A", "S014": "A",
    "S003": "C", "S004": "C", "S008": "B", "S015": "A",
}


@dataclass(frozen=True)
class RouteVariant:
    aircraft_type: str
    order: tuple[str, ...]
    energy_kwh: float
    duration_s: float
    delivery_offset_s: tuple[tuple[str, float], ...]

    def delivery_offsets(self) -> dict[str, float]:
        return dict(self.delivery_offset_s)


@dataclass
class ScheduledTrip:
    trip_id: str
    box_ids: tuple[str, ...]
    variant: RouteVariant
    drone_id: str
    battery_id: str
    start_s: float
    return_s: float
    end_soc: float
    charge_complete_s: float


def hard_deadline(box: dict) -> float | None:
    """医疗物资按期望时刻硬约束；首批箱按首批截止时刻硬约束。"""
    if "-MED-" in box["id"]:
        return box["expected_time_s"]
    if box["is_first_batch"]:
        return box["first_batch_deadline_s"]
    return None


class Q2Model:
    def __init__(self):
        self.aircraft = load_aircraft_types()
        self.boxes = load_boxes()
        self.box_by_id = {b["id"]: b for b in self.boxes}
        self.nodes = load_nodes()
        dem, lat, lon = load_dem()
        self.terrain = Terrain(dem, lat, lon)
        self.segments = build_all_segments(self.terrain, self.nodes)
        self.drones, self.battery_stock, self.full_charge_s = load_fleet()
        self.drones_by_type = {
            g: tuple(d for d, t in self.drones if t == g) for g in self.aircraft
        }
        self.batteries_by_type = {
            g: tuple(f"{g}-BAT-{i:02d}" for i in range(1, n + 1))
            for g, n in self.battery_stock.items()
        }

    @lru_cache(maxsize=200000)
    def variants(self, box_ids_key: tuple[str, ...]) -> tuple[RouteVariant, ...]:
        boxes = [self.box_by_id[x] for x in box_ids_key]
        stops = tuple(sorted({b["service"] for b in boxes}))
        if not stops or len(stops) > MAX_STOPS:
            return ()
        total_mass = sum(b["mass_kg"] for b in boxes)
        total_volume = sum(b["volume_m3"] for b in boxes)
        by_stop = {s: [b for b in boxes if b["service"] == s] for s in stops}
        variants: list[RouteVariant] = []

        for g, ac in self.aircraft.items():
            if total_mass > ac.max_payload_kg + 1e-9 or total_volume > ac.volume_m3 + 1e-12:
                continue
            energy_limit = (1.0 - ac.reserve_ratio) * ac.battery_kwh
            initial = ac.setup_time_s + len(boxes) * ac.load_time_per_box_s
            for order in itertools.permutations(stops):
                payload = total_mass
                energy = 0.0
                elapsed = initial
                delivery: dict[str, float] = {}
                src = "O01"
                feasible = True
                for stop in order:
                    seg = self.segments[(src, stop)]
                    energy += segment_energy_kwh(ac, payload, seg)
                    if energy > energy_limit + 1e-10:
                        feasible = False
                        break
                    elapsed += seg.flight_time_s(ac)
                    stop_boxes = by_stop[stop]
                    elapsed += ac.handover_base_s + ac.handover_per_box_s * len(stop_boxes)
                    for b in stop_boxes:
                        delivery[b["id"]] = elapsed
                    payload -= sum(b["mass_kg"] for b in stop_boxes)
                    src = stop
                if not feasible:
                    continue
                back = self.segments[(src, "O01")]
                energy += segment_energy_kwh(ac, 0.0, back)
                elapsed += back.flight_time_s(ac)
                if energy <= energy_limit + 1e-10:
                    variants.append(RouteVariant(
                        g, order, energy, elapsed, tuple(sorted(delivery.items()))
                    ))

        # 同机型、同顺序只会有一个结果；排序保证随机搜索可复现。
        variants.sort(key=lambda v: (v.aircraft_type, v.duration_s, v.energy_kwh, v.order))
        return tuple(variants)

    def feasible(self, box_ids: list[str] | tuple[str, ...]) -> bool:
        return bool(self.variants(tuple(sorted(box_ids))))

    def feasible_types(self, box_ids: list[str] | tuple[str, ...]) -> set[str]:
        return {v.aircraft_type for v in self.variants(tuple(sorted(box_ids)))}

    def construct_service_batches(self, rng: random.Random) -> list[tuple[str, ...]]:
        """按服务区组批，作为零迟到目标的可行性优先起点。"""
        by_service: dict[str, list[dict]] = {}
        for box in self.boxes:
            by_service.setdefault(box["service"], []).append(box)

        batches: list[list[str]] = []
        for service in sorted(by_service):
            hard_boxes = [b for b in by_service[service] if hard_deadline(b) is not None]
            soft_boxes = [b for b in by_service[service] if hard_deadline(b) is None]
            for group in (hard_boxes, soft_boxes):
                group.sort(key=lambda b: (
                    hard_deadline(b) if hard_deadline(b) is not None else b["expected_time_s"],
                    -b["priority"],
                    rng.random(),
                ))
                service_batches: list[list[str]] = []
                for box in group:
                    candidates = []
                    for index, batch in enumerate(service_batches):
                        trial = batch + [box["id"]]
                        variants = self.variants(tuple(sorted(trial)))
                        if variants:
                            best = min(variants, key=lambda v: (v.duration_s, v.energy_kwh))
                            candidates.append((best.duration_s, best.energy_kwh, index))
                    if candidates:
                        _, _, index = min(candidates)
                        service_batches[index].append(box["id"])
                    else:
                        service_batches.append([box["id"]])
                batches.extend(service_batches)
        return [tuple(sorted(batch)) for batch in batches]

    def construct_batches(self, rng: random.Random,
                          isolate_hard: bool = False) -> list[tuple[str, ...]]:
        """随机化顺序插入：只要能放入已有架次就不新开架次。"""
        hard_by_service: dict[str, list[str]] = {}
        remaining = []
        for b in self.boxes:
            deadline = hard_deadline(b)
            if deadline is not None and deadline <= 7200.0:
                hard_by_service.setdefault(b["service"], []).append(b["id"])
            else:
                remaining.append(b)

        def key(b: dict):
            due = hard_deadline(b)
            due_rank = due if due is not None else (b["expected_time_s"] or 1e9) + 1800.0
            # 截止时刻主导，随机扰动让多次构造探索不同组合。
            return due_rank + rng.uniform(-1200.0, 1200.0), -b["mass_kg"]

        remaining.sort(key=key)
        # 3600 s 首波与 7200 s 第二波分别形成并行骨架；10800 s 硬时限货箱
        # 仍开放跨区组批，由事件调度检查截止时刻。
        protected_services = sorted(hard_by_service)
        batches: list[list[str]] = [sorted(hard_by_service[s]) for s in protected_services]
        protected_types = [PROTECTED_WAVE_TYPE[s] for s in protected_services]
        for box in remaining:
            candidates = []
            for i, batch in enumerate(batches):
                trial = batch + [box["id"]]
                variants = self.variants(tuple(sorted(trial)))
                if not variants:
                    continue
                # 向首批骨架填充普通物资时，不牺牲其原有机型选择，避免挤占 C 型资源。
                if isolate_hard and i < len(protected_types):
                    trial_types = {v.aircraft_type for v in variants}
                    if protected_types[i] not in trial_types:
                        continue
                    variants = tuple(v for v in variants if v.aircraft_type == protected_types[i])
                    hard_ids = [x for x in batch if hard_deadline(self.box_by_id[x]) is not None]
                    if any(
                        any(
                            v.delivery_offsets()[x]
                            > hard_deadline(self.box_by_id[x]) + 1e-7
                            for x in hard_ids
                        )
                        for v in variants
                    ):
                        variants = tuple(
                            v for v in variants
                            if all(
                                v.delivery_offsets()[x]
                                <= hard_deadline(self.box_by_id[x]) + 1e-7
                                for x in hard_ids
                            )
                        )
                    if not variants:
                        continue
                existing_stops = {self.box_by_id[x]["service"] for x in batch}
                new_stop_penalty = 0.8 if box["service"] not in existing_stops else 0.0
                best = min(variants, key=lambda v: v.energy_kwh + v.duration_s / 10000.0)
                candidates.append((best.energy_kwh + best.duration_s / 10000.0
                                   + new_stop_penalty, i))
            if not candidates:
                batches.append([box["id"]])
                continue
            candidates.sort()
            # 从最好的少数插入位置中随机选取，兼顾紧凑装载与搜索多样性。
            pool = candidates[:min(3, len(candidates))]
            _, chosen = pool[0] if rng.random() < 0.72 else rng.choice(pool)
            batches[chosen].append(box["id"])

        return [tuple(sorted(x)) for x in batches]

    def improve_batches(self, batches: list[tuple[str, ...]], rng: random.Random,
                        attempts: int = 80, allow_cross_service: bool = True
                        ) -> list[tuple[str, ...]]:
        """合并与两架次重分配，优先减少架次并改善可行装载。"""
        work = [list(x) for x in batches]
        for _ in range(attempts):
            if len(work) < 2:
                break
            i, j = rng.sample(range(len(work)), 2)
            if i > j:
                i, j = j, i
            if not allow_cross_service:
                services = {
                    self.box_by_id[x]["service"] for x in work[i] + work[j]
                }
                if len(services) > 1:
                    continue
            # 首批骨架承担硬时限，不在架次压缩阶段相互合并或重分配。
            if any(hard_deadline(self.box_by_id[x]) is not None for x in work[i] + work[j]):
                continue
            pool = work[i] + work[j]
            if self.feasible(pool):
                work[i] = pool
                del work[j]
                continue

            rng.shuffle(pool)
            left: list[str] = []
            right: list[str] = []
            ok = True
            for bid in pool:
                choices = []
                if self.feasible(left + [bid]):
                    choices.append(left)
                if self.feasible(right + [bid]):
                    choices.append(right)
                if not choices:
                    ok = False
                    break
                target = min(choices, key=len) if rng.random() < 0.7 else rng.choice(choices)
                target.append(bid)
            if ok and left and right:
                work[i], work[j] = left, right
        return [tuple(sorted(x)) for x in work]

    def schedule(self, batches: list[tuple[str, ...]], rng: random.Random) -> tuple[list[ScheduledTrip], dict]:
        drone_ready = {d: 0.0 for d, _ in self.drones}
        battery_ready = {
            b: 0.0 for batteries in self.batteries_by_type.values() for b in batteries
        }

        def urgency(batch: tuple[str, ...]) -> tuple[float, float]:
            bs = [self.box_by_id[x] for x in batch]
            hard = [hard_deadline(b) for b in bs if hard_deadline(b) is not None]
            due = min(hard) if hard else min(b["expected_time_s"] for b in bs)
            return due + rng.uniform(-300.0, 300.0), -sum(b["priority"] for b in bs)

        pending = sorted(batches, key=urgency)
        scheduled: list[ScheduledTrip] = []
        deliveries: dict[str, float] = {}

        for k, batch in enumerate(pending, 1):
            choices = []
            route_variants = self.variants(tuple(sorted(batch)))
            for variant in route_variants:
                g = variant.aircraft_type
                offsets = variant.delivery_offsets()
                for drone in self.drones_by_type[g]:
                    for battery in self.batteries_by_type[g]:
                        start = max(drone_ready[drone], battery_ready[battery])
                        hard_count = 0
                        hard_tardy = 0.0
                        soft_tardy = 0.0
                        for bid in batch:
                            b = self.box_by_id[bid]
                            delivered = start + offsets[bid]
                            deadline = hard_deadline(b)
                            if deadline is not None:
                                late = max(0.0, delivered - deadline)
                                hard_count += int(late > 1e-7)
                                hard_tardy += late
                            else:
                                soft_tardy += b["priority"] * max(
                                    0.0, delivered - b["expected_time_s"]
                                )
                        finish = start + variant.duration_s
                        choices.append(((hard_count, hard_tardy, soft_tardy, finish,
                                         variant.energy_kwh), variant, drone, battery, start))
            if not choices:
                raise RuntimeError(f"架次无可用机型：{batch}")
            choices.sort(key=lambda x: x[0])
            # 同等级选择保留少量随机性，以探索不同资源时间线。
            best_key = choices[0][0]
            near = [c for c in choices if c[0][:3] == best_key[:3]][:6]
            chosen = near[0] if rng.random() < 0.85 else rng.choice(near)
            _, variant, drone, battery, start = chosen
            finish = start + variant.duration_s
            end_soc = 1.0 - variant.energy_kwh / self.aircraft[variant.aircraft_type].battery_kwh
            charge_done = finish + charge_time_s(
                self.full_charge_s[variant.aircraft_type], end_soc
            )
            trip = ScheduledTrip(
                f"Q2-{k:03d}", tuple(sorted(batch)), variant, drone, battery,
                start, finish, end_soc, charge_done
            )
            scheduled.append(trip)
            drone_ready[drone] = finish
            battery_ready[battery] = charge_done
            offsets = variant.delivery_offsets()
            for bid in batch:
                deliveries[bid] = start + offsets[bid]

        hard_late = []
        weighted_tardiness = 0.0
        late_boxes = 0
        for b in self.boxes:
            t = deliveries[b["id"]]
            deadline = hard_deadline(b)
            if deadline is not None and t > deadline + 1e-7:
                hard_late.append((b["id"], t - deadline))
            if deadline is None:
                late = max(0.0, t - b["expected_time_s"])
                weighted_tardiness += b["priority"] * late
                late_boxes += int(late > 1e-7)

        metrics = {
            "hard_late_count": len(hard_late),
            "hard_tardiness_s": sum(x[1] for x in hard_late),
            "soft_late_count": late_boxes,
            "weighted_tardiness": weighted_tardiness,
            "makespan_s": max(t.return_s for t in scheduled),
            "energy_kwh": sum(t.variant.energy_kwh for t in scheduled),
            "sorties": len(scheduled),
            "deliveries": deliveries,
        }
        return scheduled, metrics


def ontime_score(metrics: dict) -> tuple:
    """先消除所有迟到，再优先缩短最后返场时间。"""
    return (
        metrics["hard_late_count"],
        metrics["hard_tardiness_s"],
        metrics["weighted_tardiness"],
        metrics["makespan_s"],
        metrics["energy_kwh"],
        metrics["sorties"],
    )


def solve(iterations: int = 4800, seed: int = 20260926,
          objective: str = "ontime") -> tuple[Q2Model, list[ScheduledTrip], dict]:
    model = Q2Model()
    master = random.Random(seed)
    best = None
    for it in range(iterations):
        rng = random.Random(master.randrange(2**63))
        if objective == "ontime":
            batches = model.construct_batches(rng, isolate_hard=True)
            batches = model.improve_batches(batches, rng)
        else:
            batches = model.construct_batches(rng)
            batches = model.improve_batches(batches, rng)
        try:
            scheduled, metrics = model.schedule(batches, rng)
            if objective == "ontime":
                for _ in range(4):
                    late_ids = {
                        b["id"] for b in model.boxes
                        if metrics["deliveries"][b["id"]] > (
                            hard_deadline(b)
                            if hard_deadline(b) is not None
                            else b["expected_time_s"]
                        ) + 1e-7
                    }
                    if not late_ids and metrics["hard_late_count"] == 0:
                        break
                    repaired = []
                    for batch in batches:
                        kept = [bid for bid in batch if bid not in late_ids]
                        if kept:
                            repaired.append(tuple(kept))
                        repaired.extend((bid,) for bid in batch if bid in late_ids)
                    batches = repaired
                    scheduled, metrics = model.schedule(batches, rng)
        except RuntimeError:
            continue
        if objective == "ontime":
            score = ontime_score(metrics)
        elif objective == "sorties":
            score = (metrics["hard_late_count"], metrics["hard_tardiness_s"],
                     metrics["sorties"], metrics["weighted_tardiness"],
                     metrics["makespan_s"], metrics["energy_kwh"])
        else:
            raise ValueError(f"unknown objective: {objective}")
        if best is None or score < best[0]:
            best = (score, scheduled, metrics)
            print(
                f"iter={it:03d} hard={metrics['hard_late_count']} "
                f"soft={metrics['soft_late_count']} sorties={metrics['sorties']} "
                f"makespan={metrics['makespan_s']:.1f} "
                f"energy={metrics['energy_kwh']:.4f} "
                f"wt={metrics['weighted_tardiness']:.1f}"
            )
    assert best is not None
    return model, best[1], best[2]


def export_results(model: Q2Model, trips: list[ScheduledTrip], metrics: dict,
                   suffix: str = "") -> None:
    os.makedirs(RESULT_DIR, exist_ok=True)
    tag = f"_{suffix}" if suffix else ""
    trip_path = os.path.join(RESULT_DIR, f"Q2_运输架次{tag}.csv")
    delivery_path = os.path.join(RESULT_DIR, f"Q2_逐箱交付{tag}.csv")
    battery_path = os.path.join(RESULT_DIR, f"Q2_电池资源台账{tag}.csv")
    summary_path = os.path.join(RESULT_DIR, f"Q2_指标汇总{tag}.csv")

    with open(trip_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["架次编号", "无人机编号", "机型编号", "电池编号", "开始时刻s",
                    "访问服务区顺序", "返回O01时刻s", "架次能耗kWh", "返航SOC%",
                    "货箱编号列表"])
        for t in sorted(trips, key=lambda x: (x.start_s, x.trip_id)):
            w.writerow([t.trip_id, t.drone_id, t.variant.aircraft_type, t.battery_id,
                        round(t.start_s, 3), ",".join(t.variant.order), round(t.return_s, 3),
                        round(t.variant.energy_kwh, 6), round(100.0 * t.end_soc, 4),
                        ",".join(t.box_ids)])

    trip_of_box = {bid: t for t in trips for bid in t.box_ids}
    with open(delivery_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["货箱编号", "架次编号", "服务区编号", "交付完成时刻s",
                    "硬时限s", "期望送达时刻s", "是否硬时限满足"])
        for b in sorted(model.boxes, key=lambda x: x["id"]):
            t = trip_of_box[b["id"]]
            delivered = metrics["deliveries"][b["id"]]
            deadline = hard_deadline(b)
            w.writerow([b["id"], t.trip_id, b["service"], round(delivered, 3),
                        "" if deadline is None else round(deadline, 3),
                        round(b["expected_time_s"], 3),
                        "" if deadline is None else ("是" if delivered <= deadline + 1e-7 else "否")])

    with open(battery_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["电池编号", "机型编号", "架次编号", "占用开始s", "任务返回s",
                    "返航SOC%", "充电完成s"])
        for t in sorted(trips, key=lambda x: (x.battery_id, x.start_s)):
            w.writerow([t.battery_id, t.variant.aircraft_type, t.trip_id,
                        round(t.start_s, 3), round(t.return_s, 3),
                        round(100.0 * t.end_soc, 4), round(t.charge_complete_s, 3)])

    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["指标", "数值"])
        for key in ("hard_late_count", "soft_late_count", "weighted_tardiness",
                    "makespan_s", "energy_kwh", "sorties"):
            value = metrics[key]
            w.writerow([key, round(value, 4) if isinstance(value, float) else value])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="求解问题二多点多架次调度")
    parser.add_argument("--objective", choices=("ontime", "sorties"), default="ontime")
    parser.add_argument("--iterations", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    model, trips, metrics = solve(args.iterations, args.seed, args.objective)
    export_results(model, trips, metrics, args.suffix)
    print("\n== Q2 recommended solution ==")
    for key in ("hard_late_count", "soft_late_count", "weighted_tardiness",
                "makespan_s", "energy_kwh", "sorties"):
        print(f"{key}: {metrics[key]}")


if __name__ == "__main__":
    main()
