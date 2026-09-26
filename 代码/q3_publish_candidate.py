"""Validate and publish staged Q3 results while retaining the previous main files."""
import csv
import os
import shutil

import q3
import q3_validate


def main():
    root = q3.RESULT_DIR
    stage = os.path.join(root, 'Q3_from_Q2')
    q3_validate.RESULT_DIR = stage
    q3_validate.main()
    def metrics(folder):
        with open(os.path.join(folder, 'Q3_指标汇总.csv'), encoding='utf-8-sig', newline='') as f:
            return {r['指标']: float(r['数值']) for r in csv.DictReader(f)}
    old, new = metrics(root), metrics(stage)
    if new['联合任务完成时间s'] >= old['联合任务完成时间s']:
        raise ValueError('Candidate is not faster; keep it as an alternative instead')
    archive = os.path.join(root, f"Q3_archive_{old['联合任务完成时间s']:.3f}s")
    os.makedirs(archive, exist_ok=True)
    names = [f'Q3_{s}.csv' for s in ('运输架次','逐箱交付','中继架次','通信保障','指标汇总')]
    names += ['Q3_transport_optimized.json','Q3_relay_optimized.json']
    for name in names+['Q3_结果.xlsx','Q3_方案对比.csv']:
        source,target = os.path.join(root,name),os.path.join(archive,name)
        if os.path.exists(source) and not os.path.exists(target):
            shutil.copy2(source,target)
    for name in names:
        shutil.copy2(os.path.join(stage,name),os.path.join(root,name))
    path = os.path.join(root,'Q3_方案对比.csv')
    with open(path,encoding='utf-8-sig',newline='') as f:
        rows=list(csv.reader(f))
    values=['新Q2驱动22架次',new['运输架次数'],new['中继架次数'],new['联合任务完成时间s'],
            new['运输能耗kWh'],new['中继能耗kWh'],new['总能耗kWh'],new['硬时限违约箱数'],new['普通物资迟到箱数']]
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f)
        writer.writerows(rows[:1]+[values]+rows[1:])
    print('Published Q3; previous result archived at',archive)


if __name__ == '__main__':
    main()
