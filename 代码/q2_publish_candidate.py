"""Validate a Q2 candidate, archive the previous main result, then publish CSVs."""
import argparse
from collections import Counter
import csv
import os
import shutil

from q2 import RESULT_DIR
from q2_validate import main as validate


def read(name):
    with open(os.path.join(RESULT_DIR, name), encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def run(suffix):
    validate(suffix, require_zero_late=True)
    metrics = {r['指标']: float(r['数值']) for r in read(f'Q2_指标汇总_{suffix}.csv')}
    old = {r['指标']: float(r['数值']) for r in read('Q2_指标汇总.csv')}
    if metrics['makespan_s'] >= old['makespan_s']:
        raise ValueError('Candidate is not faster than the published result')
    archive = os.path.join(RESULT_DIR, f"Q2_archive_{old['makespan_s']:.3f}s")
    os.makedirs(archive, exist_ok=True)
    names = ['运输架次', '逐箱交付', '电池资源台账', '指标汇总']
    for name in [f'Q2_{n}.csv' for n in names] + ['Q2_结果.xlsx', 'Q2_方案对比.csv']:
        source = os.path.join(RESULT_DIR, name)
        target = os.path.join(archive, name)
        if os.path.exists(source) and not os.path.exists(target):
            shutil.copy2(source, target)
    for name in names:
        shutil.copy2(os.path.join(RESULT_DIR, f'Q2_{name}_{suffix}.csv'),
                     os.path.join(RESULT_DIR, f'Q2_{name}.csv'))
    comparison = read('Q2_方案对比.csv')
    fields = list(comparison[0])
    comparison[0]['方案'] = '初始22架次方案（存档）'
    for tag, label in [(suffix, '时效推荐方案'), ('fast_candidate', '22架次低能耗对照')]:
        rows = read(f'Q2_运输架次_{tag}.csv')
        values = {r['指标']: r['数值'] for r in read(f'Q2_指标汇总_{tag}.csv')}
        types = Counter(r['机型编号'] for r in rows)
        record = [label, values['hard_late_count'], values['soft_late_count'], values['weighted_tardiness'],
            values['makespan_s'], values['energy_kwh'], len(rows), types['A'], types['B'], types['C'],
            sum(',' in r['访问服务区顺序'] for r in rows), len({r['电池编号'] for r in rows})]
        comparison.insert(0 if tag == suffix else 1, dict(zip(fields, record)))
    with open(os.path.join(RESULT_DIR, 'Q2_方案对比.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison)
    print('Published', suffix, metrics, 'archive', archive)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('suffix')
    run(parser.parse_args().suffix)
