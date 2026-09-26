"""Restricted route-pool selection followed by exact resource scheduling.

Equivalent boxes share counts only when site, mass, volume and effective
zero-lateness deadline agree. A restricted-pool bound is never global.
"""
import argparse
from collections import Counter, defaultdict
import itertools
import json
import math
import os

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from q2 import Q2Model, RESULT_DIR, hard_deadline, export_results
from q2_cp_schedule import solve as schedule


class Pool:
    def __init__(self, model):
        self.model = model
        grouped = defaultdict(list)
        for b in model.boxes:
            deadline = min(b['expected_time_s'], hard_deadline(b) or math.inf)
            grouped[(b['service'], b['mass_kg'], b['volume_m3'], deadline)].append(b['id'])
        self.keys = sorted(grouped)
        self.ids = [sorted(grouped[k]) for k in self.keys]
        self.class_of = {b: i for i, ids in enumerate(self.ids) for b in ids}
        self.patterns = {}

    def counts(self, batch):
        return tuple(sorted(Counter(self.class_of[b] for b in batch).items()))

    def add(self, counts):
        counts = tuple((i, n) for i, n in counts if n)
        if not counts or counts in self.patterns:
            return
        batch = tuple(sorted(b for i, n in counts for b in self.ids[i][:n]))
        feasible = []
        for v in self.model.variants(batch):
            latest = min(self.keys[self.class_of[b]][3] - off
                         for b, off in v.delivery_offset_s)
            if latest >= 0:
                feasible.append((v, latest))
        if feasible:
            self.patterns[counts] = feasible

    def generate(self, neighbors):
        sites = sorted({k[0] for k in self.keys})
        singles = {}
        for site in sites:
            indices = [i for i, k in enumerate(self.keys) if k[0] == site]
            patterns = []
            for numbers in itertools.product(*(range(len(self.ids[i]) + 1) for i in indices)):
                counts = tuple((i, n) for i, n in zip(indices, numbers) if n)
                if counts and sum(self.keys[i][1]*n for i, n in counts) <= 80:
                    self.add(counts)
                    if counts in self.patterns:
                        patterns.append(counts)
            singles[site] = patterns
        pairs = set()
        for a in sites:
            nearby = sorted((b for b in sites if b != a),
                            key=lambda b: self.model.segments[a, b].horizontal_m)
            pairs.update(tuple(sorted((a, b))) for b in nearby[:neighbors])
        for a, b in sorted(pairs):
            for left in singles[a]:
                for right in singles[b]:
                    counts = tuple(sorted(left + right))
                    if sum(self.keys[i][1]*n for i, n in counts) <= 80:
                        self.add(counts)
            print('pool', a, b, len(self.patterns), flush=True)
        with open(os.path.join(RESULT_DIR, 'Q2_fast_candidate.json'), encoding='utf-8') as f:
            baseline = json.load(f)
        for batch in baseline['state']['batches']:
            self.add(self.counts(batch))
        return baseline

    def restore(self, selected):
        available = [list(ids) for ids in self.ids]
        batches = []
        for counts, copies in selected:
            for _ in range(copies):
                batch = []
                for i, n in counts:
                    assert len(available[i]) >= n
                    batch.extend(available[i][:n])
                    del available[i][:n]
                batches.append(tuple(sorted(batch)))
        assert not any(available) and len(batches) == 22
        return batches


def run(args):
    model = Q2Model()
    pool = Pool(model)
    baseline = pool.generate(args.neighbors)
    print('patterns', len(pool.patterns), 'classes', len(pool.keys), flush=True)
    columns = []
    for counts, variants in pool.patterns.items():
        maximum = min(len(pool.ids[i]) // n for i, n in counts)
        for g in 'ABC':
            options = [v for v, _ in variants if v.aircraft_type == g]
            if options:
                # Only duration matters in this relaxation. Scheduling reopens all orders.
                columns.append((counts, min(options, key=lambda v: v.duration_s), maximum))
    n = len(columns)
    size = len(pool.keys)
    rr, cc, data = [], [], []
    for j, (counts, v, _) in enumerate(columns):
        for i, amount in counts:
            rr.append(i); cc.append(j); data.append(amount)
        rr.extend([size, size+1+'ABC'.index(v.aircraft_type)])
        cc.extend([j, j]); data.extend([1, v.duration_s])
    for k, g in enumerate('ABC'):
        rr.append(size+1+k); cc.append(n); data.append(-len(model.drones_by_type[g]))
    lo = [len(ids) for ids in pool.ids] + [22] + [-np.inf]*3
    hi = [len(ids) for ids in pool.ids] + [22] + [0]*3
    objective = np.zeros(n+1)
    objective[-1] = 1
    bounds = Bounds(np.zeros(n+1), [maximum for _, _, maximum in columns]+[9000])
    log = dict(seed=args.seed, neighbors=args.neighbors, patterns=len(pool.patterns),
               classes=len(pool.keys), target_s=args.target, iterations=[])
    best = float(baseline['score'][0])
    previous = os.path.join(RESULT_DIR, 'Q2_pool_candidate.json')
    if os.path.exists(previous):
        with open(previous, encoding='utf-8') as f:
            best = min(best, json.load(f)['metrics']['makespan_s'])
    for iteration in range(args.rounds):
        matrix = coo_matrix((data, (rr, cc)), shape=(len(lo), n+1)).tocsc()
        solver = milp(objective, integrality=[1]*n+[0], bounds=bounds,
                      constraints=LinearConstraint(matrix, lo, hi),
                      options=dict(time_limit=args.master_seconds, mip_rel_gap=.002))
        item = dict(iteration=iteration, status=solver.message,
                    restricted_master_bound_s=getattr(solver, 'mip_dual_bound', None),
                    bound_scope='initial pool' if iteration == 0 else 'pool with diversification cuts')
        if solver.x is None:
            log['iterations'].append(item)
            break
        values = np.rint(solver.x[:n]).astype(int)
        selected_counts = Counter()
        for (p, _, _), value in zip(columns, values):
            selected_counts[p] += int(value)
        selected = [(p, value) for p, value in selected_counts.items() if value]
        batches = pool.restore(selected)
        item['workload_s'] = float(solver.x[-1])
        item['batches'] = batches
        item['assigned_work_s'] = {g: sum(v.duration_s*value
            for (_, v, _), value in zip(columns, values) if v.aircraft_type == g) for g in 'ABC'}
        print('master', item['workload_s'], item['assigned_work_s'], flush=True)
        result = schedule(model, batches, args.schedule_seconds, max_stops=3)
        if result:
            trips, metrics, lower, scheduling_status = result
            item.update(makespan_s=metrics['makespan_s'], energy_kwh=metrics['energy_kwh'],
                        scheduling_status=scheduling_status, fixed_batch_bound_s=lower)
            print('scheduled', metrics['makespan_s'], metrics['energy_kwh'], flush=True)
            if metrics['makespan_s'] < best - 1e-6:
                best = metrics['makespan_s']
                export_results(model, trips, metrics, 'pool_candidate')
                with open(os.path.join(RESULT_DIR, 'Q2_pool_candidate.json'), 'w', encoding='utf-8') as f:
                    json.dump(dict(metrics={k: v for k, v in metrics.items() if k != 'deliveries'},
                                   batches=batches, seed=args.seed), f, indent=2)
        else:
            item['scheduling_status'] = 'NO_SOLUTION_WITHIN_LIMIT'
        log['iterations'].append(item)
        with open(os.path.join(RESULT_DIR, 'Q2_pool_experiment.json'), 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2)
        # Diversify support, not an exhaustive Benders cut: bounds after this cut
        # describe the further-restricted search, not the original candidate pool.
        support = {p for p, _ in selected}
        for j, (p, _, _) in enumerate(columns):
            if p in support:
                rr.append(len(lo)); cc.append(j); data.append(1)
        lo.append(-np.inf); hi.append(21)
        if best < args.target:
            break
    with open(os.path.join(RESULT_DIR, 'Q2_pool_experiment.json'), 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2)
    print('best', best, 'target', args.target, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--neighbors', type=int, default=3)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--master-seconds', type=float, default=30)
    parser.add_argument('--schedule-seconds', type=float, default=20)
    parser.add_argument('--target', type=float, default=6000)
    parser.add_argument('--seed', type=int, default=20261007)
    run(parser.parse_args())
