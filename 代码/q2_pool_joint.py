"""Joint pattern selection and interval scheduling in a restricted route pool."""
import argparse
from collections import defaultdict
import csv
import json
import math
import os
import random

from q2_pool_search import Pool
from q2_cp_schedule import cp_model
from q2 import Q2Model, RESULT_DIR, ScheduledTrip, export_results
from core import charge_time_s


def run(args):
    model = Q2Model()
    pool = Pool(model)
    with open(os.path.join(RESULT_DIR, 'Q2_pool_experiment.json'), encoding='utf-8') as f:
        experiment = json.load(f)
    batches = []
    for iteration in experiment['iterations']:
        batches.extend(iteration.get('batches', []))
    with open(os.path.join(RESULT_DIR, 'Q2_fast_candidate.json'), encoding='utf-8') as f:
        batches.extend(json.load(f)['state']['batches'])
    with open(os.path.join(RESULT_DIR, f'Q2_运输架次_{args.source}.csv'), encoding='utf-8-sig') as f:
        baseline = list(csv.DictReader(f))
    batches.extend(r['货箱编号列表'].split(',') for r in baseline)
    for batch in batches:
        pool.add(pool.counts(batch))
    rng = random.Random(args.seed)
    baseline_patterns = [pool.counts(r['货箱编号列表'].split(',')) for r in baseline]
    for _ in range(args.samples):
        a, b = rng.sample(baseline_patterns, 2)
        total = defaultdict(int)
        for i, n in a+b:
            total[i] += n
        if len({pool.keys[i][0] for i in total}) > 3:
            continue
        left = tuple((i, rng.randint(0, n)) for i, n in sorted(total.items()))
        right = tuple((i, total[i]-n) for i, n in left)
        pool.add(left)
        pool.add(right)

    cp = cp_model.CpModel()
    horizon = math.ceil(max(float(r['返回O01时刻s']) for r in baseline)*1000)+5
    finish = cp.new_int_var(0, horizon, 'makespan')
    cover = [[] for _ in pool.keys]
    drone = {g: [] for g in 'ABC'}
    battery = {g: [] for g in 'ABC'}
    workload = {g: [] for g in 'ABC'}
    choices = []
    hint_rows = defaultdict(list)
    for row in baseline:
        hint_rows[pool.counts(row['货箱编号列表'].split(','))].append(row)
    for j, (pattern, variants) in enumerate(pool.patterns.items()):
        maximum = min(len(pool.ids[i]) // n for i, n in pattern)
        previous = None
        previous_start = None
        for copy in range(maximum):
            start = cp.new_int_var(0, horizon, f's{j}_{copy}')
            active = cp.new_bool_var(f'a{j}_{copy}')
            if previous is not None:
                cp.add(active <= previous)
                cp.add(start >= previous_start).only_enforce_if(active)
            previous, previous_start = active, start
            local = []
            hint = hint_rows[pattern][copy] if copy < len(hint_rows[pattern]) else None
            cp.add_hint(start, round(float(hint['开始时刻s'])*1000) if hint else 0)
            cp.add_hint(active, int(hint is not None))
            cp.add(start == 0).only_enforce_if(active.Not())
            for k, (v, latest) in enumerate(variants):
                g = v.aircraft_type
                present = cp.new_bool_var(f'p{j}_{copy}_{k}')
                duration = math.ceil(v.duration_s*1000)
                soc = 1-v.energy_kwh/model.aircraft[g].battery_kwh
                busy = math.ceil((v.duration_s+charge_time_s(model.full_charge_s[g], soc))*1000)
                drone[g].append(cp.new_optional_fixed_size_interval_var(start, duration, present, f'd{j}_{copy}_{k}'))
                battery[g].append(cp.new_optional_fixed_size_interval_var(start, busy, present, f'b{j}_{copy}_{k}'))
                cp.add(start <= math.floor(latest*1000)).only_enforce_if(present)
                cp.add(finish >= start+duration).only_enforce_if(present)
                cp.add_hint(present, int(bool(hint) and hint['机型编号'] == g
                    and tuple(hint['访问服务区顺序'].split(',')) == v.order))
                local.append(present)
                choices.append((pattern, v, start, present))
                workload[g].append(duration*present)
            cp.add(sum(local) == active)
            for i, n in pattern:
                cover[i].append(n*active)
    for i, terms in enumerate(cover):
        cp.add(sum(terms) == len(pool.ids[i]))
    cp.add(sum(p for _, _, _, p in choices) == 22)
    for g in 'ABC':
        capacity = len(model.drones_by_type[g])
        cp.add_cumulative(drone[g], [1]*len(drone[g]), capacity)
        cp.add_cumulative(battery[g], [1]*len(battery[g]), model.battery_stock[g])
        cp.add(sum(workload[g]) <= capacity*finish)
    cp.add_hint(finish, horizon-5)
    cp.minimize(finish)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = args.seconds
    solver.parameters.num_search_workers = 8
    solver.parameters.random_seed = args.seed
    print('joint patterns', len(pool.patterns), 'choices', len(choices), flush=True)

    class Progress(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            print('incumbent', self.objective_value/1000, flush=True)

    status = solver.solve(cp, Progress())
    print('status', solver.status_name(status), 'restricted_bound_s', solver.best_objective_bound/1000, flush=True)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return
    selected = sorted(((solver.value(s)/1000, p, v) for p, v, s, x in choices if solver.value(x)),
                      key=lambda row: (row[0], row[1], row[2].aircraft_type, row[2].order))
    available = [list(ids) for ids in pool.ids]
    dr = {d: 0. for d, _ in model.drones}
    br = {b: 0. for ids in model.batteries_by_type.values() for b in ids}
    trips = []
    for start, pattern, old_variant in selected:
        batch = []
        for i, n in pattern:
            assert len(available[i]) >= n
            batch.extend(available[i][:n])
            del available[i][:n]
        batch = tuple(sorted(batch))
        v = next(v for v in model.variants(batch) if v.aircraft_type == old_variant.aircraft_type
                 and v.order == old_variant.order)
        assert abs(v.duration_s-old_variant.duration_s) < 1e-8
        assert abs(v.energy_kwh-old_variant.energy_kwh) < 1e-8
        g = v.aircraft_type
        d = next(d for d in model.drones_by_type[g] if dr[d] <= start+1e-8)
        b = next(b for b in model.batteries_by_type[g] if br[b] <= start+1e-8)
        end = start+v.duration_s
        soc = 1-v.energy_kwh/model.aircraft[g].battery_kwh
        charge = end+charge_time_s(model.full_charge_s[g], soc)
        trips.append(ScheduledTrip(f'Q2-{len(trips)+1:03d}', batch, v, d, b, start, end, soc, charge))
        dr[d] = end
        br[b] = charge
    assert not any(available) and len(trips) == 22
    metrics = dict(hard_late_count=0, soft_late_count=0, weighted_tardiness=0,
        makespan_s=max(t.return_s for t in trips), energy_kwh=sum(t.variant.energy_kwh for t in trips),
        sorties=len(trips), deliveries={x: t.start_s+off for t in trips for x, off in t.variant.delivery_offset_s})
    export_results(model, trips, metrics, 'joint_pool_candidate')
    with open(os.path.join(RESULT_DIR, 'Q2_joint_pool_candidate.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(status=solver.status_name(status), restricted_bound_s=solver.best_objective_bound/1000,
            seed=args.seed, samples=args.samples, seconds=args.seconds, source=args.source,
            patterns=len(pool.patterns), metrics=metrics), f, indent=2)
    print('result', metrics['makespan_s'], metrics['energy_kwh'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=120)
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('--seed', type=int, default=20261008)
    parser.add_argument('--source', default='pool_candidate')
    run(parser.parse_args())
