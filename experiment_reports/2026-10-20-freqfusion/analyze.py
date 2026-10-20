"""Compare the archived full SVRDD runs without running inference or retraining."""

import csv
import json
import math
from pathlib import Path
from statistics import mean

import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = {
    'baseline': ROOT / 'experiment_reports/2026-10-17/rtdetr-resnet18',
    'scsa': ROOT / 'experiment_reports/2026-10-17-scsa-enabled/rtdetr-resnet18-SCSA-enabled-s0',
    'freqfusion': HERE / 'rtdetr-resnet18-FreqFusion-P3-s0',
    'scsa_freqfusion': HERE / 'rtdetr-resnet18-SCSA-FreqFusion-P3-s0',
}
METRICS = ('metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)',
           'metrics/mAP50-95(B)', 'train/giou_loss', 'train/cls_loss', 'train/l1_loss',
           'val/giou_loss', 'val/cls_loss', 'val/l1_loss')
PROTOCOL = ('epochs', 'batch', 'imgsz', 'optimizer', 'lr0', 'lrf', 'cos_lr',
            'weight_decay', 'warmup_epochs', 'close_mosaic', 'seed', 'amp',
            'deterministic', 'cache', 'workers', 'nbs')


def read_run(path):
    with (path / 'results.csv').open(newline='', encoding='utf-8') as handle:
        rows = [{key.strip(): float(value) for key, value in row.items()}
                for row in csv.DictReader(handle)]
    assert len(rows) == 300 and [int(row['epoch']) for row in rows] == list(range(1, 301)), path
    assert all(math.isfinite(value) for row in rows for value in row.values()), path
    best = max(rows, key=lambda row: row['metrics/mAP50-95(B)'])
    args = yaml.safe_load((path / 'args.yaml').read_text(encoding='utf-8'))
    return {
        'csv': str((path / 'results.csv').relative_to(ROOT)),
        'args': {key: args.get(key) for key in PROTOCOL},
        'best_epoch': int(best['epoch']),
        'best': {key: best[key] for key in METRICS},
        'last': {key: rows[-1][key] for key in METRICS},
        'mean_last_20': {key: mean(row[key] for row in rows[-20:]) for key in METRICS},
        'selected_epochs': {str(epoch): {
            'map50': rows[epoch - 1]['metrics/mAP50(B)'],
            'map50_95': rows[epoch - 1]['metrics/mAP50-95(B)']}
            for epoch in (100, 150, 200, 250, 280, 290, 300)},
        'checkpoint_gap_map50_minus_map50_95': (
            best['metrics/mAP50(B)'] - best['metrics/mAP50-95(B)']),
    }, rows


def main():
    summaries = {}
    rows = {}
    for name, path in RUNS.items():
        summaries[name], rows[name] = read_run(path)
    score = 'metrics/mAP50-95(B)'
    ap50 = 'metrics/mAP50(B)'
    delta = {}
    for name in ('scsa', 'freqfusion', 'scsa_freqfusion'):
        delta[name] = {key: (summaries[name]['best'][key] - summaries['baseline']['best'][key]) * 100
                       for key in (score, ap50)}
    combo_vs_scsa = {key: (summaries['scsa_freqfusion']['best'][key] - summaries['scsa']['best'][key]) * 100
                     for key in (score, ap50)}
    combo_vs_freqfusion = {key: (summaries['scsa_freqfusion']['best'][key] - summaries['freqfusion']['best'][key]) * 100
                           for key in (score, ap50)}
    interaction = {key: (summaries['scsa_freqfusion']['best'][key] - summaries['scsa']['best'][key]
                         - summaries['freqfusion']['best'][key] + summaries['baseline']['best'][key]) * 100
                   for key in (score, ap50)}
    late_wins = {name: {key: sum(rows[name][epoch - 1][key] > rows['baseline'][epoch - 1][key]
                                 for epoch in range(201, 301)) for key in (score, ap50)}
                 for name in ('scsa', 'freqfusion', 'scsa_freqfusion')}
    report = {
        'scope': 'Archived validation CSV only; no fresh validation, test-set measurement, or class-wise AP50-95.',
        'selection': 'Maximum recorded mAP50-95 across 300 contiguous epochs.',
        'runs': summaries,
        'delta_vs_baseline_percentage_points': delta,
        'combo_vs_scsa_percentage_points': combo_vs_scsa,
        'combo_vs_freqfusion_percentage_points': combo_vs_freqfusion,
        'factorial_interaction_percentage_points': interaction,
        'wins_vs_baseline_epochs_201_to_300': late_wins,
        'protocol_differences': {key: {name: summaries[name]['args'][key] for name in RUNS}
                                 for key in PROTOCOL if len({summaries[name]['args'][key] for name in RUNS}) > 1},
    }
    (HERE / 'analysis.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    colors = {'baseline': '#292929', 'scsa': '#3366cc',
              'freqfusion': '#e69500', 'scsa_freqfusion': '#c53030'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = ((score, 1, 300, 'mAP50-95: all epochs'),
              (ap50, 1, 300, 'mAP50: all epochs'),
              (score, 201, 300, 'mAP50-95: last 100 epochs'),
              (ap50, 201, 300, 'mAP50: last 100 epochs'))
    for ax, (key, start, end, title) in zip(axes.flat, panels):
        for name, item in rows.items():
            window = item[start - 1:end]
            ax.plot([row['epoch'] for row in window], [row[key] for row in window],
                    label=name.replace('_', '+'), color=colors[name], linewidth=1.5)
        ax.set(title=title, xlabel='Epoch', ylabel='Validation AP')
        ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.98))
    fig.subplots_adjust(top=0.89, bottom=0.08, left=0.08, right=0.98, hspace=0.34, wspace=0.19)
    fig.savefig(HERE / 'comparison_curves.png', dpi=180)
    plt.close(fig)
    for name, item in summaries.items():
        print(name, item['best_epoch'], item['best'][ap50], item['best'][score],
              item['mean_last_20'][score], item['best']['val/giou_loss'], item['best']['val/l1_loss'])
    print('delta_pp', delta)
    print('combo_vs_scsa_pp', combo_vs_scsa)
    print('combo_vs_freqfusion_pp', combo_vs_freqfusion)
    print('interaction_pp', interaction)
    print('late_wins', late_wins)
    print('protocol_differences', report['protocol_differences'])


if __name__ == '__main__':
    main()
