"""Fixed-sortie Q2 search with free batching, ordering and aircraft assignment."""
import argparse
import csv
import json
import math
import os
import random
from functools import lru_cache

from core import charge_time_s
from q2 import Q2Model, ScheduledTrip, RESULT_DIR, export_results, hard_deadline


class FastSearch:
    def __init__(self):
        self.model=Q2Model()

    @lru_cache(200000)
    def options(self,batch):
        if len({self.model.box_by_id[x]['service'] for x in batch})>3:
            return ()
        return tuple((v,min((hard_deadline(self.model.box_by_id[x]) or self.model.box_by_id[x]['expected_time_s'])-off
                           for x,off in v.delivery_offset_s)) for v in self.model.variants(batch))

    def schedule(self,state):
        batches,priority,bias=state
        m=self.model
        drones={d:0. for d,_ in m.drones}
        batteries={b:0. for bs in m.batteries_by_type.values() for b in bs}
        trips=[]
        for j in sorted(range(len(batches)),key=lambda j:priority[j]):
            options=[]
            for v,latest in self.options(batches[j]):
                g=v.aircraft_type
                d=min(m.drones_by_type[g],key=lambda d:drones[d])
                b=min(m.batteries_by_type[g],key=lambda b:batteries[b])
                start=max(drones[d],batteries[b])
                if start>latest+1e-7:continue
                options.append((start+v.duration_s+bias[j].get(g,0),v.energy_kwh,v,d,b,start))
            if not options:return None
            _,_,v,d,b,start=min(options,key=lambda x:x[:2])
            end=start+v.duration_s
            soc=1-v.energy_kwh/m.aircraft[v.aircraft_type].battery_kwh
            charge=end+charge_time_s(m.full_charge_s[v.aircraft_type],soc)
            trips.append(ScheduledTrip(f'Q2-{len(trips)+1:03d}',batches[j],v,d,b,start,end,soc,charge))
            drones[d]=end;batteries[b]=charge
        return (round(max(drones.values()),6),round(sum(t.variant.energy_kwh for t in trips),9)),trips

    def run(self,iterations,seed):
        rng=random.Random(seed)
        candidate=os.path.join(RESULT_DIR,'Q2_fast_candidate.json')
        if os.path.exists(candidate):
            with open(candidate,encoding='utf-8') as f:record=json.load(f)
            raw=record['state'];state=([tuple(b) for b in raw['batches']],raw['priority'],raw['bias'])
        else:
            with open(os.path.join(RESULT_DIR,'Q2_运输架次.csv'),encoding='utf-8-sig',newline='') as f:
                rows=list(csv.DictReader(f))
            state=([tuple(sorted(r['货箱编号列表'].split(','))) for r in rows],list(range(len(rows))),
                   [{g:(-10000 if g==r['机型编号'] else 0) for g in 'ABC'} for r in rows])
        assert len(state[0])==22
        current=self.schedule(state)
        assert current is not None
        best=(current[0],state,current[1])
        print('initial',best[0],flush=True)
        for it in range(iterations):
            if it%3000==0:
                state=best[1];current=(best[0],best[2])
            batches,priority,bias=state
            batches=list(batches);priority=list(priority);bias=[dict(x) for x in bias]
            i,j=rng.sample(range(22),2)
            action=rng.randrange(6)
            if action==0:
                priority[i]+=rng.uniform(-15,15)
            elif action==1:
                bias[i][rng.choice('ABC')]=rng.uniform(-2500,2500)
            elif action in (2,3):
                if rng.random()<.85:
                    sites={self.model.box_by_id[x]['service'] for x in batches[i]}
                    js=[k for k,b in enumerate(batches) if k!=i and sites & {self.model.box_by_id[x]['service'] for x in b}]
                    if not js:continue
                    j=rng.choice(js)
                pool=list(batches[i]+batches[j]);rng.shuffle(pool)
                if action==2:
                    n=rng.randrange(1,len(pool));left,right=pool[:n],pool[n:]
                else:
                    left=list(batches[i]);right=list(batches[j])
                    a=rng.choice(left);b=rng.choice(right)
                    left.remove(a);right.remove(b);left.append(b);right.append(a)
                left,right=tuple(sorted(left)),tuple(sorted(right))
                if not self.options(left) or not self.options(right):continue
                batches[i],batches[j]=left,right
            elif action==4:
                priority[i],priority[j]=priority[j],priority[i]
            else:
                merged=tuple(sorted(batches[i]+batches[j]))
                if not self.options(merged):continue
                ks=[k for k,b in enumerate(batches) if k not in (i,j) and len(b)>1]
                if not ks:continue
                k=rng.choice(ks);pool=list(batches[k]);rng.shuffle(pool);n=rng.randrange(1,len(pool))
                left,right=tuple(sorted(pool[:n])),tuple(sorted(pool[n:]))
                if not self.options(left) or not self.options(right):continue
                batches[i],batches[j],batches[k]=merged,left,right
                priority[j]=priority[k]+rng.uniform(-4,4)
                bias[j]={g:0 for g in 'ABC'}
            trial=(batches,priority,bias)
            result=self.schedule(trial)
            if result is None:continue
            if result[0]<best[0]:
                best=(result[0],trial,result[1])
                print('best',it,best[0],flush=True)
                self.save(best,seed,it)
            temperature=220*(1-(it%3000)/3000)+2
            def search_cost(item):
                return item[0][0]+0.05*sum(t.return_s for t in item[1])/22
            if result[0]<current[0] or rng.random()<math.exp(min(0,(search_cost(current)-search_cost(result))/temperature)):
                state,current=trial,result
        self.save(best,seed,iterations)

    def save(self,best,seed,iteration):
        score,state,trips=best
        metrics=dict(hard_late_count=0,soft_late_count=0,weighted_tardiness=0,
                     makespan_s=score[0],energy_kwh=score[1],sorties=len(trips),
                     deliveries={x:t.start_s+off for t in trips for x,off in t.variant.delivery_offset_s})
        export_results(self.model,trips,metrics,'fast_candidate')
        with open(os.path.join(RESULT_DIR,'Q2_fast_candidate.json'),'w',encoding='utf-8') as f:
            json.dump(dict(score=score,seed=seed,iteration=iteration,
                           state=dict(batches=state[0],priority=state[1],bias=state[2])),f,indent=2)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--iterations',type=int,default=200000)
    parser.add_argument('--seed',type=int,default=20261001)
    parser.add_argument('--replay')
    args=parser.parse_args()
    search=FastSearch()
    if args.replay:
        with open(args.replay,encoding='utf-8') as f:record=json.load(f)
        raw=record['state'];state=([tuple(b) for b in raw['batches']],raw['priority'],raw['bias'])
        result=search.schedule(state)
        assert result is not None and all(abs(a-b)<1e-7 for a,b in zip(result[0],record['score']))
        print('Replay passed',result[0])
    else:
        search.run(args.iterations,args.seed)
