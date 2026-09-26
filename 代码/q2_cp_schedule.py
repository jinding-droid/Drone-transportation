"""CP-SAT scheduling of a fixed batch partition, using conservative millisecond ticks."""
import argparse
import csv
import json
import math
import os
import sys

from q2 import Q2Model, ScheduledTrip, RESULT_DIR, export_results, hard_deadline
from core import charge_time_s

sys.path.insert(0,os.path.join(os.path.dirname(RESULT_DIR),'.codex-tmp-q2-xlsx','solverdeps'))
from ortools.sat.python import cp_model


def solve(model,batches,seconds=20,max_stops=2):
    cp=cp_model.CpModel()
    horizon=12000000
    drone_intervals={g:[] for g in 'ABC'}
    battery_intervals={g:[] for g in 'ABC'}
    choices=[]
    finish=cp.new_int_var(0,horizon,'makespan')
    for i,batch in enumerate(batches):
        start=cp.new_int_var(0,horizon,f'start{i}')
        selected=[]
        for j,v in enumerate(model.variants(tuple(sorted(batch)))):
            if len(v.order)>max_stops:continue
            g=v.aircraft_type
            latest=min(min(hard_deadline(model.box_by_id[x]) or math.inf,model.box_by_id[x]['expected_time_s'])-off for x,off in v.delivery_offset_s)
            present=cp.new_bool_var(f'p{i}_{j}')
            duration=math.ceil(v.duration_s*1000)
            recharge=charge_time_s(model.full_charge_s[g],1-v.energy_kwh/model.aircraft[g].battery_kwh)
            busy=math.ceil((v.duration_s+recharge)*1000)
            drone_intervals[g].append(cp.new_optional_fixed_size_interval_var(start,duration,present,f'd{i}_{j}'))
            battery_intervals[g].append(cp.new_optional_fixed_size_interval_var(start,busy,present,f'b{i}_{j}'))
            cp.add(start<=math.floor(latest*1000)).only_enforce_if(present)
            cp.add(finish>=start+duration).only_enforce_if(present)
            selected.append(present);choices.append((batch,v,start,present))
        cp.add_exactly_one(selected)
    for g in 'ABC':
        cp.add_cumulative(drone_intervals[g],[1]*len(drone_intervals[g]),len(model.drones_by_type[g]))
        cp.add_cumulative(battery_intervals[g],[1]*len(battery_intervals[g]),model.battery_stock[g])
    cp.minimize(finish)
    solver=cp_model.CpSolver()
    solver.parameters.max_time_in_seconds=seconds
    solver.parameters.num_search_workers=8
    solver.parameters.random_seed=20261001
    status=solver.solve(cp)
    if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):return None
    rows=sorted([(solver.value(s)/1000,batch,v) for batch,v,s,p in choices if solver.value(p)],key=lambda x:x[0])
    dr={d:0. for d,_ in model.drones}
    br={b:0. for bs in model.batteries_by_type.values() for b in bs}
    trips=[]
    for start,batch,v in rows:
        g=v.aircraft_type
        d=next(d for d in model.drones_by_type[g] if dr[d]<=start+1e-8)
        b=next(b for b in model.batteries_by_type[g] if br[b]<=start+1e-8)
        end=start+v.duration_s
        soc=1-v.energy_kwh/model.aircraft[g].battery_kwh
        charge=end+charge_time_s(model.full_charge_s[g],soc)
        trips.append(ScheduledTrip(f'Q2-{len(trips)+1:03d}',batch,v,d,b,start,end,soc,charge))
        dr[d]=end;br[b]=charge
    metrics=dict(hard_late_count=0,soft_late_count=0,weighted_tardiness=0,
                 makespan_s=max(t.return_s for t in trips),energy_kwh=sum(t.variant.energy_kwh for t in trips),
                 sorties=len(trips),deliveries={x:t.start_s+off for t in trips for x,off in t.variant.delivery_offset_s})
    return trips,metrics,solver.best_objective_bound/1000,solver.status_name(status)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seconds',type=float,default=45)
    args=parser.parse_args()
    model=Q2Model()
    with open(os.path.join(RESULT_DIR,'Q2_fast_candidate.json'),encoding='utf-8') as f:record=json.load(f)
    result=solve(model,[tuple(b) for b in record['state']['batches']],args.seconds)
    if result is None:raise RuntimeError('No feasible solution found within time limit')
    trips,metrics,bound,status=result
    export_results(model,trips,metrics,'cp_candidate')
    print(status,'makespan',metrics['makespan_s'],'energy',metrics['energy_kwh'],'fixed-batch bound',bound)
