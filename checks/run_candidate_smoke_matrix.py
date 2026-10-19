"""Run two real-data epochs for every unique ablation in both candidates.

Requires the tiny local sample from checks/download_svrdd_smoke.py. This is
an integration check, not an accuracy experiment. Training output is ignored
by Git under runs/.
"""

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import EXPERIMENTS


DATA = ROOT / 'datasets/svrdd-smoke-hf/data.yaml'
PROJECT = ROOT / 'runs/smoke-matrix'
LOGS = ROOT / 'runs/smoke-matrix-logs'
NAMES = (
    'baseline', 'scsa', 'freqfusion', 'scale-decoder', 'scsa-freqfusion',
    'scsa-scale-decoder', 'freqfusion-scale-decoder', 'scsa-freqfusion-scale-decoder',
    'dysample', 'fdconv-c4', 'scsa-dysample', 'scsa-fdconv-c4',
    'dysample-fdconv-c4', 'scsa-dysample-fdconv-c4',
)


def completed_epochs(rows):
    if len(rows) != 2:
        return False
    normalized = [{key.strip(): value for key, value in row.items()} for row in rows]
    losses = ('train/giou_loss', 'train/cls_loss', 'train/l1_loss',
              'val/giou_loss', 'val/cls_loss', 'val/l1_loss')
    return ([int(float(row['epoch'])) for row in normalized] == [1, 2] and
            all(math.isfinite(float(row[key])) for row in normalized for key in losses))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true', help='Reuse completed two-epoch runs')
    options = parser.parse_args()
    if not DATA.is_file():
        raise FileNotFoundError(f'Run checks/download_svrdd_smoke.py first: {DATA}')
    LOGS.mkdir(parents=True, exist_ok=True)
    PROJECT.mkdir(parents=True, exist_ok=True)
    results = []
    for index, name in enumerate(NAMES, 1):
        run_dir = PROJECT / EXPERIMENTS[name]['name']
        cmd = [sys.executable, 'train.py', '--experiment', name, '--data', str(DATA),
               '--project', str(PROJECT), '--epochs', '2', '--batch', '2', '--imgsz', '256',
               '--workers', '0', '--warmup-iterations', '0', '--no-plots', '--device', '0']
        print(f'[{index}/{len(NAMES)}] {name}', flush=True)
        log_path = LOGS / f'{name}.log'
        csv_path = run_dir / 'results.csv'
        rows = list(csv.DictReader(csv_path.open(encoding='utf-8'))) if csv_path.is_file() else []
        complete = (completed_epochs(rows) and (run_dir / 'weights/last.pt').is_file() and
                    (run_dir / 'source_snapshot/provenance.json').is_file())
        if options.resume and complete:
            returncode = 0
            print('  reusing completed run', flush=True)
        else:
            with log_path.open('w', encoding='utf-8') as log:
                result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
            returncode = result.returncode
            rows = list(csv.DictReader(csv_path.open(encoding='utf-8'))) if csv_path.is_file() else []
            complete = (completed_epochs(rows) and (run_dir / 'weights/last.pt').is_file() and
                        (run_dir / 'source_snapshot/provenance.json').is_file())
        success = returncode == 0 and complete
        item = {'experiment': name, 'exit_code': returncode, 'epochs_logged': len(rows),
                'last_weight': str(run_dir / 'weights/last.pt'), 'log': str(log_path), 'passed': success}
        results.append(item)
        print(f'  passed={success} exit={returncode} epochs={len(rows)}', flush=True)
        if not success:
            tail = '\n'.join(log_path.read_text(encoding='utf-8', errors='replace').splitlines()[-18:])
            print(tail.encode('ascii', errors='backslashreplace').decode('ascii'),
                  flush=True)
    (LOGS / 'summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Completed: {sum(x["passed"] for x in results)}/{len(results)} passed', flush=True)
    return 0 if all(x['passed'] for x in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
