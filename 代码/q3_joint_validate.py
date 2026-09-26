"""Stage and validate the joint-search candidate without replacing the baseline."""
import csv
import json
import os
import shutil

import q3
import q3_validate


def main():
    result = q3.RESULT_DIR
    stage = os.path.join(result, 'Q3_joint_validated')
    os.makedirs(stage, exist_ok=True)
    with open(os.path.join(result,'Q3_joint_candidate.json'),encoding='utf-8') as f:
        candidate=json.load(f)
    with open(os.path.join(stage,'Q3_relay_optimized.json'),'w',encoding='utf-8') as f:
        json.dump(candidate['relays'],f,indent=2)
    with open(os.path.join(result,'Q2_运输架次_Q3_joint_candidate.csv'),encoding='utf-8-sig',newline='') as f:
        trips=list(csv.DictReader(f))
    for trip in trips:
        trip.update(start_s=float(trip['开始时刻s']), return_s=float(trip['返回O01时刻s']),
                    route=tuple(trip['访问服务区顺序'].split(',')),
                    box_ids=tuple(trip['货箱编号列表'].split(',')))
    with open(os.path.join(stage,'Q3_transport_optimized.json'),'w',encoding='utf-8') as f:
        json.dump(trips,f,ensure_ascii=False,indent=2)
    q3.RESULT_DIR=stage
    link=q3.LinkModel()
    missions=q3.build_relay_missions(link)
    phases=q3.transport_phases(trips)
    for step in (0.5,0.1):
        total,missing=q3.verify_coverage(phases,link,missions,step)
        print('joint coverage',step,total,missing,flush=True)
        if missing:
            raise ValueError('Candidate has coverage failures')
    q3.write_q3_results(trips,phases,link,missions)
    q3_validate.RESULT_DIR=stage
    q3_validate.main()
    print('Validated candidate stored at',stage)


if __name__=='__main__':
    main()
