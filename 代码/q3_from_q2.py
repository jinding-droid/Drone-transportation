"""Reschedule the current Q2 routes jointly with three relay service windows."""
import argparse
import csv
import json
import math
import os
import random
from dataclasses import replace

import q3
import q3_validate
from q3_joint_search import JointSearch
from q2_cp_schedule import cp_model
from core import charge_time_s
from q2 import hard_deadline


def relocate(search, trips):
    samples = q3.sample_phases(q3.transport_phases(trips), search.link, 0.5)
    points = list(dict.fromkeys(s.point for s in samples if not s.direct))
    rng = random.Random(20260926)
    masks = [[search.link.available(p,r.hover,'access') for p in points] for r in search.relays]
    all_candidates = [r.hover for r in search.relays]
    for i in (2,1,0):
        essential = [p for j,p in enumerate(points) if not any(masks[k][j] for k in range(3) if k!=i)]
        rng.shuffle(essential)
        old = search.relays[i].hover
        candidates = [old]
        for agl in (225., 299.999, 180., 150.):
            candidates.append(q3.Point3D(old.lon, old.lat, round(search.link.terrain.elevation(old.lon, old.lat)+agl, 3)))
        for _ in range(2000):
            lon = round(old.lon+rng.uniform(-.02,.02),7)
            lat = round(old.lat+rng.uniform(-.02,.02),7)
            agl = rng.choice([299.99,225.,150.])
            candidates.append(q3.Point3D(lon,lat,round(search.link.terrain.elevation(lon,lat)+agl,3)))
        for point in candidates:
            if not search.link.available(point, search.link.gateway, 'backhaul'):
                continue
            all_candidates.append(point)
            if all(search.link.available(p,point,'access') for p in essential):
                search.relays[i] = replace(search.relays[i],hover=point)
                search.profiles[i] = q3.relay_profile(point,search.link)
                print('relocated relay',i,point,'essential points',len(essential),flush=True)
                return
    all_candidates = list(dict.fromkeys(all_candidates))
    rng.shuffle(all_candidates)
    all_candidates = list(dict.fromkeys([r.hover for r in search.relays]+all_candidates[:900]))
    dense_points = points
    points = points[::max(1,len(points)//500)]
    print('joint spatial search',len(points),len(all_candidates),flush=True)
    cover = [[] for _ in points]
    cp = cp_model.CpModel()
    selected = [cp.new_bool_var(f'point{i}') for i in range(len(all_candidates))]
    early = [cp.new_bool_var(f'early{i}') for i in range(len(all_candidates))]
    for a,b in zip(early,selected):
        cp.add(a<=b)
    cp.add(sum(early)==2)
    urgent_sites={b['service'] for b in search.model.boxes
                  if min(b['expected_time_s'],hard_deadline(b) or math.inf)<=3600}
    for site in urgent_sites:
        node=search.model.nodes[site]
        target=q3.Point3D(node.lon,node.lat,node.op_alt_m)
        if not search.link.available(target,search.link.gateway,'direct'):
            cp.add(sum(x for p,x in zip(all_candidates,early) if search.link.available(target,p,'access'))>=1)
    for i,p in enumerate(all_candidates):
        for j,target in enumerate(points):
            if search.link.available(target,p,'access'):
                cover[j].append(selected[i])
    for terms in cover:
        cp.add(sum(terms)>=1)
    cp.add(sum(selected)==3)
    solver=cp_model.CpSolver()
    solver.parameters.max_time_in_seconds=20
    solver.parameters.num_search_workers=8
    for refinement in range(12):
        status=solver.solve(cp)
        if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):
            raise RuntimeError('No three-point cover found in sampled spatial pool')
        chosen=[p for p,x in zip(all_candidates,selected) if solver.value(x)]
        missing=[p for p in dense_points if not any(search.link.available(p,c,'access') for c in chosen)]
        print('spatial refinement',refinement,'missing',len(missing),flush=True)
        if not missing:
            break
        for target in missing[::max(1,len(missing)//40)]:
            cp.add(sum(x for p,x in zip(all_candidates,selected) if search.link.available(target,p,'access'))>=1)
    else:
        raise RuntimeError('Spatial refinement did not converge')
    early_points=[p for p,x in zip(all_candidates,early) if solver.value(x)]
    chosen=sorted(early_points,key=lambda p:p.lon)+[p for p in chosen if p not in early_points]
    for i,p in enumerate(chosen):
        search.relays[i]=replace(search.relays[i],hover=p)
        search.profiles[i]=q3.relay_profile(p,search.link)
    print('joint locations',chosen,flush=True)


def main(seconds, repair=False, repair_spatial=False, relay_config=None):
    search = JointSearch()
    model = search.model
    if relay_config:
        with open(relay_config,encoding='utf-8') as f:
            config=json.load(f)
        for i,r in enumerate(config):
            point=q3.Point3D(r['lon'],r['lat'],r['alt_m'])
            search.relays[i]=replace(search.relays[i],hover=point)
            search.profiles[i]=q3.relay_profile(point,search.link)
    trips = q3.load_q2_trips()
    if repair or repair_spatial:
        by_id={t['架次编号']:t for t in trips}
        moves=[('S002-MED-01','Q2-003','Q2-011'),('S012-MED-01','Q2-004','Q2-012')]
        if not repair_spatial:
            moves.append(('S007-HYG-01','Q2-016','Q2-021'))
        for box,source,target in moves:
            a,b=by_id[source],by_id[target]
            assert box in a['box_ids'] and box not in b['box_ids']
            a['box_ids']=tuple(x for x in a['box_ids'] if x!=box)
            b['box_ids']=tuple(sorted(b['box_ids']+(box,)))
        for row in trips:
            row['route']=tuple(s for s in row['route'] if any(model.box_by_id[x]['service']==s for x in row['box_ids']))
            row['货箱编号列表']=','.join(row['box_ids'])
        for row in trips:
            row['route']=tuple(sorted({model.box_by_id[x]['service'] for x in row['box_ids']}))
        print('repaired',len(moves),'box assignments; still 22 batches',flush=True)
        if repair_spatial:
            for row in trips:
                variants=model.variants(tuple(sorted(row['box_ids'])))
                v=min(variants,key=lambda v:(v.aircraft_type!=row['机型编号'],v.duration_s))
                row['机型编号']=v.aircraft_type
                row['route']=v.order
                row['return_s']=row['start_s']+v.duration_s
            if not relay_config:
                relocate(search,trips)
    else:
        relocate(search, trips)
    cp = cp_model.CpModel()
    horizon = 12000000
    finish = cp.new_int_var(0, horizon, 'finish')
    ready = [cp.new_int_var(math.ceil(p['ready_offset_s']*1000), horizon, f'ready{i}')
             for i, p in enumerate(search.profiles)]
    end = [cp.new_int_var(0, horizon, f'end{i}') for i in range(3)]
    power = search.params['hover_power_kw']+search.params['comm_extra_power_kw']
    for i, profile in enumerate(search.profiles):
        max_window = (search.params['battery_kwh']*(1-search.params['reserve_ratio_pct']/100)
                      - profile['base_energy_kwh'])*3600/power
        cp.add(end[i] >= ready[i])
        cp.add(end[i]-ready[i] <= math.floor(max_window*1000))
        cp.add(finish >= end[i]+math.ceil(profile['inbound_s']*1000))
    relay_intervals=[]
    for i,p in enumerate(search.profiles):
        departure=cp.new_int_var(0,horizon,f'relay_start{i}')
        release=cp.new_int_var(0,horizon*2,f'relay_release{i}')
        duration=cp.new_int_var(0,horizon*2,f'relay_duration{i}')
        cp.add(departure==ready[i]-math.ceil(p['ready_offset_s']*1000))
        cp.add(release==end[i]+math.ceil((p['inbound_s']+search.params['turnaround_s'])*1000))
        relay_intervals.append(cp.new_interval_var(departure,duration,release,f'relay{i}'))
    cp.add_cumulative(relay_intervals,[1]*3,2)
    drone = {g: [] for g in 'ABC'}
    battery = {g: [] for g in 'ABC'}
    records = []
    old_windows = [[] for _ in range(3)]
    for j, row in enumerate(trips):
        options = [(v, ranges, latest) for v, ranges, latest in search.choices(tuple(sorted(row['box_ids'])))
                   if repair or repair_spatial or v.aircraft_type == row['机型编号']]
        if not options:
            raise ValueError(f"Route not covered by relay locations: {row['架次编号']}")
        start = cp.new_int_var(0,horizon,f'start{j}')
        selection=[]
        for k,(v,ranges,latest) in enumerate(options):
            present=cp.new_bool_var(f'route{j}_{k}')
            selection.append(present)
            latest = min(min(model.box_by_id[x]['expected_time_s'],
                     hard_deadline(model.box_by_id[x]) or math.inf)-off
                     for x, off in v.delivery_offset_s)
            cp.add(start<=math.floor(latest*1000)).only_enforce_if(present)
            duration = math.ceil(v.duration_s*1000)
            g = v.aircraft_type
            recharge = charge_time_s(model.full_charge_s[g], 1-v.energy_kwh/model.aircraft[g].battery_kwh)
            drone[g].append(cp.new_optional_fixed_size_interval_var(start, duration,present, f'd{j}_{k}'))
            battery[g].append(cp.new_optional_fixed_size_interval_var(start, math.ceil((v.duration_s+recharge)*1000),present, f'b{j}_{k}'))
            cp.add(finish >= start+duration).only_enforce_if(present)
            for i, (lo, hi) in ranges.items():
                cp.add(ready[i] <= start+math.floor(lo*1000)).only_enforce_if(present)
                cp.add(end[i] >= start+math.ceil(hi*1000)).only_enforce_if(present)
                old_windows[i].append((row['start_s']+lo, row['start_s']+hi))
            records.append((row, v, start,present))
        cp.add_exactly_one(selection)
        cp.add_hint(start, round(row['start_s']*1000))
    print('Q2 relay demand windows', [(min(a for a,b in w), max(b for a,b in w)) for w in old_windows], flush=True)
    for g in 'ABC':
        cp.add_cumulative(drone[g], [1]*len(drone[g]), len(model.drones_by_type[g]))
        cp.add_cumulative(battery[g], [1]*len(battery[g]), model.battery_stock[g])
    cp.minimize(finish)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = 8
    solver.parameters.random_seed = 20260926
    status = solver.solve(cp)
    print('status', solver.status_name(status), 'bound', solver.best_objective_bound/1000, flush=True)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError('No feasible schedule within this fixed-location model/time limit')
    best_finish = solver.value(finish)
    primary_solver = solver
    cp.add(finish <= best_finish)
    cp.minimize(sum(round(v.energy_kwh*3600000000)*p for _,v,_,p in records)
                +round(power*1000)*sum(end[i]-ready[i] for i in range(3)))
    cp.clear_hints()
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8
    solver.parameters.random_seed = 20260926
    solver.parameters.max_time_in_seconds = min(seconds, 30)
    second = solver.solve(cp)
    if second not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        solver = primary_solver
    dr = {d: 0. for d, _ in model.drones}
    br = {b: 0. for ids in model.batteries_by_type.values() for b in ids}
    updated = []
    for row, v, variable, _ in sorted((r for r in records if solver.value(r[3])), key=lambda r: solver.value(r[2])):
        row = dict(row)
        start = solver.value(variable)/1000
        g = v.aircraft_type
        d = next(d for d in model.drones_by_type[g] if dr[d] <= start+1e-8)
        b = next(b for b in model.batteries_by_type[g] if br[b] <= start+1e-8)
        returned = start+v.duration_s
        dr[d] = returned
        br[b] = returned+charge_time_s(model.full_charge_s[g],1-v.energy_kwh/model.aircraft[g].battery_kwh)
        row.update(start_s=start, return_s=returned,route=v.order)
        row.update({'开始时刻s': f'{start:.3f}', '返回O01时刻s': f'{returned:.3f}',
                    '无人机编号': d, '电池编号': b,'机型编号':g,'访问服务区顺序':','.join(v.order),
                    '架次能耗kWh':f'{v.energy_kwh:.6f}','返航SOC%':f'{100*(1-v.energy_kwh/model.aircraft[g].battery_kwh):.4f}'})
        updated.append(row)
    config = [dict(mission_id=r.mission_id, drone_id=r.drone_id, component_id=r.component_id,
                   lon=r.hover.lon, lat=r.hover.lat, alt_m=r.hover.alt_m,
                   ready=solver.value(ready[i])/1000, service_end=solver.value(end[i])/1000)
              for i, r in enumerate(search.relays)]
    relay_free={'R01':0.,'R02':0.}
    for i in sorted(range(3),key=lambda i:config[i]['ready']-search.profiles[i]['ready_offset_s']):
        departure=config[i]['ready']-search.profiles[i]['ready_offset_s']
        aircraft=next(d for d,t in relay_free.items() if t<=departure+1e-8)
        config[i]['drone_id']=aircraft
        relay_free[aircraft]=config[i]['service_end']+search.profiles[i]['inbound_s']+search.params['turnaround_s']
    stage = os.path.join(q3.RESULT_DIR, 'Q3_from_Q2')
    os.makedirs(stage, exist_ok=True)
    for name, data in [('Q3_transport_optimized.json', updated), ('Q3_relay_optimized.json', config)]:
        with open(os.path.join(stage, name), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    q3.RESULT_DIR = stage
    missions = q3.build_relay_missions(search.link)
    phases = q3.transport_phases(updated)
    for step in (0.5, 0.1):
        total, missing = q3.verify_coverage(phases, search.link, missions, step)
        print('coverage', step, total, missing, flush=True)
        if missing:
            raise RuntimeError('Coverage verification failed; candidate not published')
    q3.write_q3_results(updated, phases, search.link, missions)
    q3_validate.RESULT_DIR = stage
    q3_validate.main()
    print('Validated candidate', stage, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--repair',action='store_true')
    parser.add_argument('--repair-spatial',action='store_true')
    parser.add_argument('--relay-config')
    args=parser.parse_args()
    main(args.seconds,args.repair,args.repair_spatial,args.relay_config)
