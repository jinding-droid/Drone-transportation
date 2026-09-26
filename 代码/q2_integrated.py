"""Joint single-site batching and scheduling with 22 active sorties using CP-SAT."""
import argparse
import math
import os
import sys
from collections import defaultdict

from q2 import Q2Model, ScheduledTrip, RESULT_DIR, export_results, hard_deadline
from core import charge_time_s, segment_energy_kwh

sys.path.insert(0,os.path.join(os.path.dirname(RESULT_DIR),'.codex-tmp-q2-xlsx','solverdeps'))
from ortools.sat.python import cp_model


def run(seconds,seed):
    m=Q2Model();cp=cp_model.CpModel()
    bysite=defaultdict(list)
    for b in m.boxes:bysite[b['service']].append(b)
    H=12000000
    intervals={g:[] for g in 'ABC'};battery_intervals={g:[] for g in 'ABC'}
    workloads={g:[] for g in 'ABC'}
    active=[];records=[]
    makespan=cp.new_int_var(0,8000000,'makespan')
    for site,boxes in bysite.items():
        count_slots=min(4,len(boxes))
        assign=[[cp.new_bool_var(f'x_{site}_{j}_{b}') for b in range(len(boxes))] for j in range(count_slots)]
        previous=None
        for j in range(count_slots):
            present=cp.new_bool_var(f'active_{site}_{j}');active.append(present)
            count=cp.new_int_var(0,len(boxes),f'count_{site}_{j}')
            mass=cp.new_int_var(0,80,f'mass_{site}_{j}')
            cp.add(count==sum(assign[j]));cp.add(count>=1).only_enforce_if(present);cp.add(count==0).only_enforce_if(present.Not())
            cp.add(mass==sum(int(b['mass_kg'])*x for b,x in zip(boxes,assign[j])))
            volume=sum(round(b['volume_m3']*1000000)*x for b,x in zip(boxes,assign[j]))
            start=cp.new_int_var(0,H,f'start_{site}_{j}')
            if previous is not None:
                cp.add(previous[0]>=present)
                cp.add(previous[1]<=start).only_enforce_if(present)
            previous=(present,start)
            choices=[]
            for g,ac in m.aircraft.items():
                select=cp.new_bool_var(f'type_{site}_{j}_{g}');choices.append(select)
                cp.add(mass<=int(ac.max_payload_kg)).only_enforce_if(select)
                cp.add(volume<=round(ac.volume_m3*1000000)).only_enforce_if(select)
                out=m.segments['O01',site];back=m.segments[site,'O01']
                fixed=ac.setup_time_s+out.flight_time_s(ac)+back.flight_time_s(ac)+ac.handover_base_s
                per=ac.load_time_per_box_s+ac.handover_per_box_s
                duration=cp.new_int_var(0,H,f'duration_{site}_{j}_{g}')
                cp.add(duration==math.ceil(fixed*1000)+round(per*1000)*count)
                workload=cp.new_int_var(0,H,f'work_{site}_{j}_{g}')
                cp.add(workload==duration).only_enforce_if(select)
                cp.add(workload==0).only_enforce_if(select.Not())
                workloads[g].append(workload)
                end=cp.new_int_var(0,2*H,f'end_{site}_{j}_{g}');cp.add(end==start+duration)
                energies=[segment_energy_kwh(ac,q,out)+segment_energy_kwh(ac,0,back) if q<=ac.max_payload_kg else ac.battery_kwh*2 for q in range(81)]
                max_mass=max(q for q,e in enumerate(energies) if e<=ac.battery_kwh*(1-ac.reserve_ratio)+1e-9)
                cp.add(mass<=max_mass).only_enforce_if(select)
                charges=[math.ceil(charge_time_s(m.full_charge_s[g],max(0,1-e/ac.battery_kwh))*1000) for e in energies]
                recharge=cp.new_int_var(0,max(charges),f'charge_{site}_{j}_{g}')
                cp.add_element(mass,charges,recharge)
                busy=cp.new_int_var(0,2*H,f'busy_{site}_{j}_{g}');cp.add(busy==duration+recharge)
                intervals[g].append(cp.new_optional_interval_var(start,duration,end,select,f'd_{site}_{j}_{g}'))
                bend=cp.new_int_var(0,3*H,f'bend_{site}_{j}_{g}');cp.add(bend==start+busy)
                battery_intervals[g].append(cp.new_optional_interval_var(start,busy,bend,select,f'b_{site}_{j}_{g}'))
                cp.add(makespan>=end).only_enforce_if(select)
                for b,x in zip(boxes,assign[j]):
                    deadline=hard_deadline(b) or b['expected_time_s']
                    cp.add(end<=math.floor((deadline+back.flight_time_s(ac))*1000)).only_enforce_if([select,x])
            cp.add(sum(choices)==present)
            records.append((site,boxes,assign[j],present,start,choices))
        for b in range(len(boxes)):cp.add_exactly_one(assign[j][b] for j in range(count_slots))
        # Identical boxes are interchangeable; ordered assignment removes label symmetry.
        categories=defaultdict(list)
        for b,box in enumerate(boxes):
            categories[(box['mass_kg'],box['volume_m3'],hard_deadline(box) or box['expected_time_s'])].append(b)
        for ids in categories.values():
            for a,b in zip(ids,ids[1:]):
                cp.add(sum(j*assign[j][a] for j in range(count_slots))<=sum(j*assign[j][b] for j in range(count_slots)))
    cp.add(sum(active)==22)
    for g in 'ABC':
        cp.add(sum(workloads[g])<=len(m.drones_by_type[g])*makespan)
        cp.add_cumulative(intervals[g],[1]*len(intervals[g]),len(m.drones_by_type[g]))
        cp.add_cumulative(battery_intervals[g],[1]*len(battery_intervals[g]),m.battery_stock[g])
    cp.minimize(makespan)
    solver=cp_model.CpSolver();solver.parameters.max_time_in_seconds=seconds
    solver.parameters.num_search_workers=8;solver.parameters.random_seed=seed
    class Progress(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            print('integrated',self.objective_value/1000,'bound',self.best_objective_bound/1000,flush=True)
    status=solver.solve(cp,Progress())
    print('status',solver.status_name(status),'bound',solver.best_objective_bound/1000,flush=True)
    if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):return
    scheduled=[]
    for site,boxes,assignment,present,start,choices in records:
        if not solver.value(present):continue
        batch=tuple(sorted(b['id'] for b,x in zip(boxes,assignment) if solver.value(x)))
        kind=next(g for g,x in zip('ABC',choices) if solver.value(x))
        v=next(v for v in m.variants(batch) if v.aircraft_type==kind)
        scheduled.append((solver.value(start)/1000,batch,v))
    dr={d:0. for d,_ in m.drones};br={b:0. for bs in m.batteries_by_type.values() for b in bs};trips=[]
    for start,batch,v in sorted(scheduled,key=lambda x:x[0]):
        g=v.aircraft_type
        d=next(d for d in m.drones_by_type[g] if dr[d]<=start+1e-8)
        b=next(b for b in m.batteries_by_type[g] if br[b]<=start+1e-8)
        end=start+v.duration_s;soc=1-v.energy_kwh/m.aircraft[g].battery_kwh
        charge=end+charge_time_s(m.full_charge_s[g],soc)
        trips.append(ScheduledTrip(f'Q2-{len(trips)+1:03d}',batch,v,d,b,start,end,soc,charge))
        dr[d]=end;br[b]=charge
    metrics=dict(hard_late_count=0,soft_late_count=0,weighted_tardiness=0,
                 makespan_s=max(t.return_s for t in trips),energy_kwh=sum(t.variant.energy_kwh for t in trips),
                 sorties=len(trips),deliveries={x:t.start_s+off for t in trips for x,off in t.variant.delivery_offset_s})
    export_results(m,trips,metrics,'integrated_candidate')
    print('result',metrics['makespan_s'],metrics['energy_kwh'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seconds',type=float,default=240)
    parser.add_argument('--seed',type=int,default=20261001);args=parser.parse_args()
    run(args.seconds,args.seed)
