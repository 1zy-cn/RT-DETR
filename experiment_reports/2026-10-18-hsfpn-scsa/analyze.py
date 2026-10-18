"""Rebuild the comparison from archived validation CSVs (no training or inference)."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
SOURCES = {
    'baseline': HERE.parent / '2026-10-17/rtdetr-resnet18',
    'hsfpn': HERE.parent / '2026-10-14/rtdetr-resnet18-HSFPN',
    'scsa': HERE.parent / '2026-10-17-scsa-enabled/rtdetr-resnet18-SCSA-enabled-s0',
    'combo': HERE / 'run',
}
METRICS = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
manifest = json.loads((HERE / 'source_manifest.json').read_text(encoding='utf-8'))
for name, meta in manifest['raw_files'].items():
    data = (HERE / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == meta['sha256'] and len(data) == meta['bytes'], name
for key, run in [('baseline', 'rtdetr-resnet18'), ('hsfpn', 'rtdetr-resnet18-HSFPN')]:
    # Older archives permit Git CRLF conversion on Windows; compare their LF form.
    data = (SOURCES[key] / 'results.csv').read_bytes().replace(b'\r\n', b'\n')
    assert hashlib.sha256(data).hexdigest() == manifest['control_csv_hashes_at_capture'][run], key

rows, args, stats = {}, {}, {}
for key, folder in SOURCES.items():
    rows[key] = [{k.strip(): float(v) for k, v in row.items()}
                 for row in csv.DictReader((folder / 'results.csv').read_text().splitlines())]
    assert [r['epoch'] for r in rows[key]] == list(range(1, 301)), key
    args[key] = yaml.safe_load((folder / 'args.yaml').read_text(encoding='utf-8'))
    best95 = max(rows[key], key=lambda r: r[METRICS[3]])
    bestfit = max(rows[key], key=lambda r: .1 * r[METRICS[2]] + .9 * r[METRICS[3]])
    stats[key] = {
        'batch': args[key]['batch'], 'epochs': len(rows[key]),
        'csv_lf_normalized_sha256': hashlib.sha256((folder / 'results.csv').read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'last': rows[key][-1], 'best_map5095_epoch': best95['epoch'],
        'best_fitness_epoch': bestfit['epoch'],
        'last20_mean_map50': statistics.mean(r[METRICS[2]] for r in rows[key][-20:]),
        'last20_mean_map5095': statistics.mean(r[METRICS[3]] for r in rows[key][-20:]),
        'epoch290_to300_map5095_pp': 100 * (rows[key][-1][METRICS[3]] - rows[key][289][METRICS[3]]),
    }
    assert best95['epoch'] == bestfit['epoch'] == 300, key

previous = json.loads((HERE.parent / '2026-10-17-scsa-enabled/analysis.json').read_text(encoding='utf-8'))
class_ap = previous['per_class_ap50_rounded']
# Manually read from this run's PR_curve.png legend, rounded to three decimals.
class_ap['combo'] = [.866, .843, .931, .828, .907, .889, .859]
deltas = {key: {m: round(100 * (stats['combo']['last'][m] - stats[key]['last'][m]), 6)
                for m in METRICS} for key in ['baseline', 'hsfpn', 'scsa']}
differences = {k: {name: arg.get(k) for name, arg in args.items()}
               for k in sorted(set().union(*(arg.keys() for arg in args.values())))
               if len({json.dumps(arg.get(k), sort_keys=True) for arg in args.values()}) > 1}
end_args = json.loads((HERE / 'checkpoint_verification.json').read_text(encoding='utf-8'))['checkpoints']['best.pt']['train_args']
summary = {
    'scope': 'Existing 300-epoch validation histories; not fresh validation or independent test evaluation.',
    'comparability': 'Combo batch=16 versus batch=8 controls: not a controlled architecture-only ablation.',
    'selection': 'All four runs have best mAP50-95 and best 0.1*mAP50+0.9*mAP50-95 at epoch 300.',
    'source_csv': {k: str(v.relative_to(HERE.parent)).replace('\\', '/') + '/results.csv' for k, v in SOURCES.items()},
    'runs': stats, 'combo_delta_percentage_points': deltas,
    'args_yaml_differences': differences,
    'combo_start_vs_checkpoint_end': {
        k: {'args_yaml': args['combo'][k], 'checkpoint_train_args': end_args[k]}
        for k in ['batch', 'nbs', 'mosaic', 'close_mosaic']},
    'runtime_mutation_note': 'Trainer sets nbs=batch before optimization and close_mosaic sets mosaic=0 for the last 10 epochs. End checkpoint args are not all initial settings.',
    'per_class_ap50_source': 'PR_curve.png legends, manually transcribed; rounded to 3 decimals, not exact per-class metrics.',
    'per_class_ap50_rounded': class_ap,
    'estimated_steps': {'training_images_at_capture': 7393, 'batch8_per_epoch': 925,
                        'batch16_per_epoch': 463, 'batch8_total_300_epochs': 277500,
                        'batch16_total_300_epochs': 138900,
                        'assumptions': 'Single device, unchanged 7393 images, no dropped final batch, accumulate=1; not measured step logs.'},
}
(HERE / 'analysis.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout='constrained')
colors = dict(zip(SOURCES, ['#3e6079', '#a17722', '#168578', '#ba4652']))
labels = {'baseline': 'Baseline (B8)', 'hsfpn': 'HSFPN (B8)', 'scsa': 'SCSA (B8)', 'combo': 'HSFPN + SCSA (B16)'}
for ax, metric, title in zip(axes[:2], METRICS[2:], ['Validation mAP50', 'Validation mAP50-95']):
    for key, records in rows.items():
        ax.plot([r['epoch'] for r in records], [r[metric] * 100 for r in records],
                color=colors[key], lw=1.4, label=labels[key], alpha=.9)
    ax.set(xlabel='Epoch', ylabel='AP (%)', title=title, xlim=(1, 300))
    ax.grid(alpha=.15)
axes[0].legend(loc='lower right', frameon=False, fontsize=9)
x = np.arange(7)
for offset, key, color in [(-.18, 'baseline', colors['baseline']), (.18, 'scsa', colors['scsa'])]:
    delta = 100 * (np.array(class_ap['combo']) - np.array(class_ap[key]))
    axes[2].bar(x + offset, delta, width=.35, color=color, label='Combo minus ' + key)
axes[2].axhline(0, lw=.8, color='#555555')
axes[2].set(xticks=x, xticklabels=class_ap['class_ids'], ylabel='AP50 difference (percentage points)',
            title='Per-class comparison (rounded PR legends)')
axes[2].legend(frameon=False, fontsize=9)
axes[2].grid(axis='y', alpha=.15)
fig.suptitle('SVRDD / ResNet18 / 300 epochs / seed 0\nCombo uses batch 16; controls use batch 8. Architecture effects are confounded.', fontsize=12)
fig.savefig(HERE / 'comparison.png', dpi=180)
fig.savefig(HERE / 'comparison.pdf')
plt.close(fig)
print(json.dumps({'runs': stats, 'deltas': deltas, 'args_differences': differences}, ensure_ascii=False, indent=2))
