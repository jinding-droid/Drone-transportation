"""Joint batch/resource search with three fixed relay locations and variable windows."""
import argparse
import csv
import json
import math
import os
import random
from functools import lru_cache

import numpy as np

from core import charge_time_s, load_relay_type
from q2 import Q2Model, ScheduledTrip, export_results, hard_deadline
from q3 import LinkModel, Point3D, Phase, build_relay_missions, relay_profile, RESULT_DIR


class JointSearch:
    def __init__(self):
        self.model = Q2Model()
        self.link = LinkModel()
        self.relays = build_relay_missions(self.link)
        self.params = load_relay_type()
        self.profiles = [relay_profile(m.hover, self.link) for m in self.relays]

    @lru_cache(None)
    def source(self, point):
        if self.link.available(point, self.link.gateway, 'direct'):
            return -1
        for i, relay in enumerate(self.relays):
            if self.link.available(point, relay.hover, 'access'):
                return i
        return 3

    @lru_cache(None)
    def leg(self, kind, src, dst):
        m = self.model
        ac, seg = m.aircraft[kind], m.segments[src, dst]
        a, b = m.nodes[src], m.nodes[dst]
        points = [Point3D(a.lon,a.lat,a.op_alt_m), Point3D(a.lon,a.lat,seg.cruise_alt_m),
                  Point3D(b.lon,b.lat,seg.cruise_alt_m), Point3D(b.lon,b.lat,b.op_alt_m)]
        durations = [seg.climb_m/ac.climb_speed_ms,seg.horizontal_m/ac.cruise_speed_ms,
                     seg.descent_m/ac.descent_speed_ms]
        ranges = {}
        now = 0.
        for p0,p1,d in zip(points,points[1:],durations):
            phase = Phase('', '', now, now+d, p0,p1)
            for time in np.linspace(now,now+d,max(2,math.ceil(d/2)+1)):
                source = self.source(phase.point(time))
                if source == 3:
                    return None
                if source >= 0:
                    lo,hi = ranges.get(source,(math.inf,-math.inf))
                    ranges[source] = (min(lo,float(time)),max(hi,float(time)))
            now += d
        return ranges

    @lru_cache(100000)
    def choices(self, batch):
        m = self.model
        out=[]
        for v in m.variants(batch):
            if len(v.order)>2:
                continue
            ac=m.aircraft[v.aircraft_type]
            now=ac.setup_time_s+len(batch)*ac.load_time_per_box_s
            ranges={}
            src='O01'
            valid=True
            for dst in v.order+('O01',):
                leg=self.leg(v.aircraft_type,src,dst)
                if leg is None:
                    valid=False;break
                for i,(a,b) in leg.items():
                    lo,hi=ranges.get(i,(math.inf,-math.inf))
                    ranges[i]=(min(lo,now+a-3),max(hi,now+b+3))
                now+=m.segments[src,dst].flight_time_s(ac)
                if dst!='O01':
                    duration=ac.handover_base_s+ac.handover_per_box_s*sum(m.box_by_id[x]['service']==dst for x in batch)
                    p=m.nodes[dst]
                    i=self.source(Point3D(p.lon,p.lat,p.op_alt_m))
                    if i==3:
                        valid=False;break
                    if i>=0:
                        lo,hi=ranges.get(i,(math.inf,-math.inf))
                        ranges[i]=(min(lo,now-3),max(hi,now+duration+3))
                    now+=duration
                src=dst
            if valid:
                latest=min((hard_deadline(m.box_by_id[x]) or m.box_by_id[x]['expected_time_s'])-offset for x,offset in v.delivery_offset_s)
                out.append((v,ranges,latest))
        return out

    def schedule(self,batches,priorities,bias,cutoff):
        m=self.model
        dr={d:0. for d,_ in m.drones}
        br={b:0. for bs in m.batteries_by_type.values() for b in bs}
        ready=[self.profiles[0]['ready_offset_s']+1,self.profiles[1]['ready_offset_s']+1,
               cutoff+self.profiles[1]['inbound_s']+self.params['turnaround_s']+self.profiles[2]['ready_offset_s']+2]
        order=sorted(range(len(batches)),key=lambda j:priorities[j])
        trips=[]
        ranges_all=[[] for _ in self.relays]
        for j in order:
            batch=batches[j]
            options=[]
            for v,ranges,latest in self.choices(batch):
                g=v.aircraft_type
                d=min(m.drones_by_type[g],key=lambda x:dr[x])
                b=min(m.batteries_by_type[g],key=lambda x:br[x])
                start=max(dr[d],br[b],*( [ready[i]-lo for i,(lo,hi) in ranges.items()] or [0.]))
                if start>latest+1e-7 or (1 in ranges and start+ranges[1][1]>cutoff):
                    continue
                end=start+v.duration_s
                options.append((end+bias[j].get(g,0),v.energy_kwh,v,d,b,start,ranges))
            if not options:
                return None
            _,_,v,d,b,start,ranges=min(options,key=lambda x:x[:2])
            end=start+v.duration_s
            soc=1-v.energy_kwh/m.aircraft[v.aircraft_type].battery_kwh
            charge=end+charge_time_s(m.full_charge_s[v.aircraft_type],soc)
            trips.append(ScheduledTrip(f'Q3-{len(trips)+1:03d}',batch,v,d,b,start,end,soc,charge))
            dr[d]=end;br[b]=charge
            for i,(lo,hi) in ranges.items():
                ranges_all[i].append((start+lo,start+hi))
        energy=0.
        relay_defs=[]
        for i,ranges in enumerate(ranges_all):
            if not ranges:
                return None
            first=min(a for a,b in ranges)
            last=max(b for a,b in ranges)
            p=self.profiles[i]
            e=p['base_energy_kwh']+(self.params['hover_power_kw']+self.params['comm_extra_power_kw'])*(last-first)/3600
            if e>self.params['battery_kwh']*(1-self.params['reserve_ratio_pct']/100):
                return None
            energy+=e
            relay_defs.append((first,last,last+p['inbound_s']))
        makespan=max(max(dr.values()),max(r[2] for r in relay_defs))
        total_energy=energy+sum(t.variant.energy_kwh for t in trips)
        return (makespan,total_energy,len(trips)),trips,relay_defs

    def run(self,iterations,seed):
        rng=random.Random(seed)
        m=self.model
        base=m.construct_fast_batches()
        candidate_path=os.path.join(RESULT_DIR,'Q2_运输架次_Q3_joint_candidate.csv')
        if os.path.exists(candidate_path):
            with open(candidate_path,encoding='utf-8-sig',newline='') as f:
                candidate_rows=list(csv.DictReader(f))
                base=[tuple(sorted(r['货箱编号列表'].split(','))) for r in candidate_rows]
        else:
            candidate_rows=None
        self.initial_batches=base
        def initialize(batches):
            pr=[]
            for batch in batches:
                due=min(hard_deadline(m.box_by_id[x]) or m.box_by_id[x]['expected_time_s'] for x in batch)
                sites={m.box_by_id[x]['service'] for x in batch}
                pr.append(due+rng.uniform(-1200,1200)-(1200 if sites & {'S010','S012','S013','S014'} else 0))
            return (batches,pr,[{g:rng.uniform(-300,300) for g in 'ABC'} for b in batches],rng.choice([3500.,3800.,4100.]))
        best=None;current=None;state=None
        def rank(score):
            return (max(0,score[0]-6999),score[1],score[0],score[2])
        if candidate_rows:
            state=(base,list(range(len(base))),
                   [{g:(-10000 if g==r['机型编号'] else 0) for g in 'ABC'} for r in candidate_rows],4300.)
            result=self.schedule(*state)
            if result is not None:
                best=(result[0],state,result);current=result[0]
                self.save(best,seed,-1)
        for it in range(iterations):
            if state is None or it%400==0:
                trial=initialize(base if best is None or rng.random()<.25 else best[1][0])
            else:
                batches,pr,bias,cutoff=state
                batches=list(batches);pr=list(pr);bias=[dict(x) for x in bias]
                action=rng.randrange(6)
                i=rng.randrange(len(batches))
                if action==0 and len(batches[i])>1 and len(batches)<24:
                    pool=list(batches[i]);rng.shuffle(pool);n=rng.randrange(1,len(pool))
                    batches[i]=tuple(sorted(pool[:n]));batches.append(tuple(sorted(pool[n:])))
                    pr.append(pr[i]+rng.uniform(-1200,1200));bias.append({g:rng.uniform(-500,500) for g in 'ABC'})
                elif action==1:
                    partners=[j for j in range(len(batches)) if j!=i and {m.box_by_id[x]['service'] for x in batches[j]}=={m.box_by_id[x]['service'] for x in batches[i]}]
                    if not partners:continue
                    j=rng.choice(partners);pool=list(batches[i]+batches[j]);rng.shuffle(pool)
                    n=rng.randrange(1,len(pool));batches[i]=tuple(sorted(pool[:n]));batches[j]=tuple(sorted(pool[n:]))
                elif action==2:
                    pr[i]+=rng.uniform(-3500,3500)
                elif action==3:
                    bias[i][rng.choice('ABC')]+=rng.uniform(-1600,1600)
                elif action==5:
                    partners=[j for j in range(len(batches)) if j!=i and {m.box_by_id[x]['service'] for x in batches[j]}=={m.box_by_id[x]['service'] for x in batches[i]}]
                    if not partners:continue
                    j=rng.choice(partners)
                    merged=tuple(sorted(batches[i]+batches[j]))
                    if not self.choices(merged):continue
                    batches[i]=merged
                    del batches[j];del pr[j];del bias[j]
                else:
                    cutoff=max(2900,min(4300,cutoff+rng.uniform(-200,200)))
                trial=(batches,pr,bias,cutoff)
            result=self.schedule(*trial)
            if result is None:continue
            score=result[0]
            if best is None or rank(score)<rank(best[0]):
                best=(score,trial,result)
                print('best',it,score,'cutoff',trial[3],flush=True)
                self.save(best,seed,it)
            temperature=60*(1-(it%400)/400)+2
            if current is None or rank(score)<rank(current) or rng.random()<math.exp(min(0,(current[0]-score[0])/temperature)):
                state,current=trial,score
        return best

    def save(self,best,seed,iteration):
        score,state,result=best
        trips=result[1]
        metrics=dict(hard_late_count=0,soft_late_count=0,weighted_tardiness=0,
                     makespan_s=max(t.return_s for t in trips),energy_kwh=sum(t.variant.energy_kwh for t in trips),
                     sorties=len(trips),deliveries={x:t.start_s+off for t in trips for x,off in t.variant.delivery_offset_s})
        export_results(self.model,trips,metrics,'Q3_joint_candidate')
        config=[dict(mission_id=r.mission_id,drone_id=r.drone_id,component_id=r.component_id,
                     lon=r.hover.lon,lat=r.hover.lat,alt_m=r.hover.alt_m,ready=w[0],service_end=w[1])
                for r,w in zip(self.relays,result[2])]
        with open(os.path.join(RESULT_DIR,'Q3_joint_candidate.json'),'w',encoding='utf-8') as f:
            json.dump(dict(score=score,relays=config,seed=seed,iteration=iteration,
                           initial_batches=self.initial_batches,
                           state=dict(batches=state[0],priorities=state[1],bias=state[2],cutoff=state[3])),f,indent=2)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--iterations',type=int,default=12000)
    parser.add_argument('--seed',type=int,default=20260927)
    parser.add_argument('--replay',help='Replay a saved candidate state without randomized search')
    args=parser.parse_args()
    search=JointSearch()
    if args.replay:
        with open(args.replay,encoding='utf-8') as f:
            record=json.load(f)
        state=record['state']
        state=([tuple(x) for x in state['batches']],state['priorities'],state['bias'],state['cutoff'])
        result=search.schedule(*state)
        assert result is not None
        assert all(abs(a-b)<1e-7 for a,b in zip(result[0],record['score']))
        print('Replay passed:',result[0])
    else:
        search.run(args.iterations,args.seed)
