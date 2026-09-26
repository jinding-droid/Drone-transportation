"""Freeze current Q3 inputs, solve/validate Q4, and archive previous outputs."""
import hashlib
import json
from pathlib import Path
import shutil

import q4
import q4_validate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(q4.RESULT_DIR)
    names = [f'Q3_{name}.csv' for name in ('运输架次', '逐箱交付', '中继架次', '通信保障', '指标汇总')]
    hashes = {name: digest(root / name) for name in names}
    version = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:12]
    stage = root / f'Q4_from_Q3_{version}'
    stage.mkdir(exist_ok=True)
    for name in names:
        shutil.copy2(root / name, stage / name)
        assert digest(stage / name) == hashes[name], 'Q3 changed while copying inputs'
    q4.RESULT_DIR = str(stage)
    q4_validate.RESULT_DIR = str(stage)
    model = q4.Q4Model()
    solutions = {k: model.solve(k) for k in (2, 3)}
    q4.write_results(model, solutions)
    q4_validate.main()
    n = len(model.components)
    record = dict(input_sha256=hashes, balance_cv_limit=q4.BALANCE_CV_LIMIT,
        relay_policy='Replicate each required full Q3 relay interval independently per group.',
        components=[model.metrics(c).sites for c in model.components],
        partition_counts={'2': 2**(n-1)-1, '3': (3**n-3*2**n+3)//6},
        inventory=model.inventory, global_peak=model.global_metrics.resources,
        objectives={str(k): model.objective(tuple(g.mask for g in groups)) for k, groups in solutions.items()})
    manifest = stage / 'Q4_input_manifest.json'
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    outputs = ['Q4_分区配置.csv', 'Q4_方案比较.csv', 'Q4_input_manifest.json']
    archive = root / f'Q4_archive_before_{version}'
    archive.mkdir(exist_ok=True)
    for name in outputs + ['Q4_结果.xlsx']:
        if (root / name).exists() and not (archive / name).exists():
            shutil.copy2(root / name, archive / name)
    if any(digest(root / name) != hashes[name] for name in names):
        raise RuntimeError('Q3 changed during Q4 solve; candidate not published')
    for name in outputs:
        shutil.copy2(stage / name, root / name)
    print('Q4 published; snapshot:', stage)
    print('partition counts:', record['partition_counts'])
    print('global resource peak:', model.global_metrics.resources)
    for k, groups in solutions.items():
        print('K=', k, 'objective=', record['objectives'][str(k)])
        for group in groups:
            print(group.sites, group.resources, 'boxes=', group.boxes, 'workload=', group.workload)


if __name__ == '__main__':
    main()
