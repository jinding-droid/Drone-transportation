"""Reproducible local relay search with fixed transport and dense coverage checks."""
import json
import math
import os
from dataclasses import replace

import numpy as np

from core import charge_time_s, load_relay_fleet, load_relay_type
from q3 import (LinkModel, Point3D, RelayMission, RESULT_DIR, build_relay_missions,
                q3_transport_trips, relay_profile, sample_phases, transport_phases,
                verify_coverage, write_q3_results)


def main():
    link = LinkModel()
    trips = q3_transport_trips()
    phases = transport_phases(trips)
    baseline = build_relay_missions(link)
    params = load_relay_type()
    full_charge = load_relay_fleet()[2]
    samples = [s for s in sample_phases(phases, link, 0.25) if not s.direct]
    assigned = [[] for _ in baseline]
    for sample in samples:
        for i, mission in enumerate(baseline):
            if (mission.link_ready_s <= sample.time_s <= mission.service_end_s
                    and link.available(sample.point, mission.hover, 'access')):
                assigned[i].append(sample)
                break
        else:
            raise ValueError('Baseline has an uncovered sample')
    rng = np.random.default_rng(20260925)
    best = list(baseline)
    transport_end = max(t['return_s'] for t in trips)
    def score(missions):
        return (max(transport_end, *(m.return_s for m in missions)),
                sum(m.energy_kwh for m in missions))
    for i, group in enumerate(assigned):
        ready = math.floor((min(s.time_s for s in group) - 1.0) * 1000) / 1000
        end = math.ceil((max(s.time_s for s in group) + 1.0) * 1000) / 1000
        points = list(dict.fromkeys(s.point for s in group))
        rng.shuffle(points)
        print('mission', i + 1, 'unique points', len(points), 'window', ready, end, flush=True)
        def evaluate(point):
            point = Point3D(round(point.lon, 7), round(point.lat, 7), round(point.alt_m, 3))
            ground = link.terrain.elevation(point.lon, point.lat)
            if not 0 <= point.alt_m - ground <= params['max_hover_agl_m']:
                return None
            if not link.available(point, link.gateway, 'backhaul'):
                return None
            profile = relay_profile(point, link)
            start = ready - profile['ready_offset_s']
            energy = profile['base_energy_kwh'] + (params['hover_power_kw'] + params['comm_extra_power_kw']) * (end-ready)/3600
            soc = 1-energy/params['battery_kwh']
            returned = end+profile['inbound_s']
            if start < 0 or soc < params['reserve_ratio_pct']/100:
                return None
            candidate = replace(best[i], start_s=start, hover=point, link_ready_s=ready,
                                service_end_s=end, return_s=returned, energy_kwh=energy,
                                end_soc=soc, charge_complete_s=returned+charge_time_s(full_charge,soc))
            trial = best.copy()
            trial[i] = candidate
            if trial[2].start_s < trial[1].return_s + params['turnaround_s']:
                return None
            if score(trial) >= score(best):
                return None
            if any(not link.available(p, point, 'access') for p in points):
                return None
            return candidate
        for radius in (0.008, 0.004, 0.002, 0.0007):
            for attempt in range(100):
                center = best[i].hover
                lon = center.lon + (0 if attempt == 0 else rng.uniform(-radius, radius))
                lat = center.lat + (0 if attempt == 0 else rng.uniform(-radius, radius))
                agl = min(299.999, max(10.0, center.alt_m-link.terrain.elevation(center.lon,center.lat)+rng.uniform(-70,40)))
                point = Point3D(lon,lat,link.terrain.elevation(lon,lat)+agl)
                candidate = evaluate(point)
                if candidate is not None:
                    best[i] = candidate
                    print('improved',i+1,score(best),flush=True)
    for step in (0.5, 0.25, 0.1):
        total, missing = verify_coverage(phases,link,best,step)
        print('verify',step,total,missing,flush=True)
        if missing:
            raise ValueError('Refined coverage failed; no results published')
    if score(best) >= score(baseline):
        raise ValueError('No improvement')
    config = [dict(mission_id=m.mission_id,drone_id=m.drone_id,component_id=m.component_id,
                   lon=m.hover.lon,lat=m.hover.lat,alt_m=m.hover.alt_m,
                   ready=m.link_ready_s,service_end=m.service_end_s) for m in best]
    with open(os.path.join(RESULT_DIR,'Q3_relay_optimized.json'),'w',encoding='utf-8') as f:
        json.dump(config,f,indent=2)
    write_q3_results(trips,phases,link,best)
    print('baseline',score(baseline),'optimized',score(best),flush=True)


if __name__ == '__main__':
    main()
