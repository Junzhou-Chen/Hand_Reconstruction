#!/usr/bin/env python3
"""Consolidate completed read-only experiments and plot, on the remote host."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--code-revision', required=True)
    args = parser.parse_args()
    if not Path('/sciclone/home/jchen57').is_dir():
        parser.error('Aggregate and plot numerical experiment results remotely')
    if args.output_dir.exists():
        raise FileExistsError('Use a fresh report directory')
    args.output_dir.mkdir(parents=True)
    base = args.experiment_dir.resolve()
    evidence = {}

    def read(relative):
        path = base / relative
        evidence[relative] = {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}
        return json.loads(path.read_text())

    driver, grad_driver = read('driver_status.json'), read('gradient_v2_status.json')
    assert driver['phase'] == grad_driver['phase'] == 'complete'
    motion = read('twm_paired/summary.json')
    assert motion['all_paired_inputs_verified']
    assert all(v['read_only_verified'] for v in motion['checkpoints'].values())
    keep = {'input/wrist_mm', 'input/root_deg', 'input/finger_deg',
            'output/wrist_mm', 'output/root_deg', 'output/finger_deg',
            'output/wrist_mm_p50', 'output/wrist_mm_p95',
            'degraded/wrist_mm', 'degraded/root_deg', 'degraded/finger_deg',
            'current/std_mm', 'current/90pct_xyz_ellipsoid_coverage',
            'future_1s/map/wrist_mm', 'future_1s/restored_hold/wrist_mm',
            'future_2s/map/wrist_mm', 'future_2s/restored_hold/wrist_mm'}
    numeric = {k: motion[k] for k in ('schema', 'training_code_version', 'manifest', 'manifest_sha256',
        'selection_counts', 'selection_sha256', 'selection_seed', 'corruption_seed', 'corruption_modes',
        'source_sha256', 'script_sha256', 'protocol', 'limitations', 'checkpoints', 'config', 'all_paired_inputs_verified')}
    numeric['by_checkpoint'] = {}
    for label, value in motion['by_checkpoint'].items():
        numeric['by_checkpoint'][label] = {'equal_source_by_hand': value['equal_source_by_hand'],
            'by_source_hand': {cell: {mode: {'samples': summary['samples'],
                'metrics': {k: v for k, v in summary['metrics'].items() if k in keep},
                'counts': {k: v for k, v in summary['counts'].items() if k in keep}}
                for mode, summary in modes.items()} for cell, modes in value['by_source_hand'].items()}}
    real, geometry, camera = {}, {}, {}
    for split in ('val', 'test'):
        real[split] = read(f'real_states_{split}/report.json')
        assert all(v['hash_before'] == v['hash_after'] for v in real[split]['models'].values())
        item = read(f'geometry_bias_{split}.json')
        geometry[split] = {k: item[k] for k in ('schema', 'source_manifest_sha256', 'script_sha256',
            'maximum_mse_identity_residual_mm2', 'reference_and_codec_audit', 'definitions', 'limitations', 'paired_source_frames')}
        geometry[split]['branches'] = {branch: {'handedness': {
            hand: {k: v for k, v in data.items() if k != 'relative_error_by_Dex21_joint'}
            for hand, data in value['handedness'].items()}} for branch, value in item['branches'].items()}
        item = read(f'camera_depth_{split}.json')
        assert item['maximum_official_get_trans_replay_error_m'] < 1e-8
        camera[split] = {k: item[k] for k in ('schema', 'source_manifest_sha256', 'script_sha256', 'formula',
            'official_hawor_revision', 'official_source_file_sha256', 'maximum_official_get_trans_replay_error_m',
            'paired_complete_states', 'limitations')}
        camera[split]['branches'] = {branch: {'handedness': {
            hand: {k: (v[:3] if k == 'largest_wrist_errors' else v) for k, v in data.items()}
            for hand, data in value['handedness'].items()}} for branch, value in item['branches'].items()}
        critical = item['branches']['last.raw']['handedness']['left']['largest_wrist_errors'][0]
        with (base / f'camera_depth_{split}_per_frame.csv').open() as handle:
            camera[split]['same_worst_raw_frame_across_branches'] = [row for row in csv.DictReader(handle)
                if row['recording'] == critical['recording'] and int(row['frame_id']) == int(critical['frame_id'])]
        camera[split]['worst_frame_selection_note'] = 'Post-hoc failure diagnosis only; all original test/val frames remain in all primary metrics.'
    gradients = {}
    for label in ('best', 'last'):
        item = read(f'gradients_{label}_v2.json')
        assert item['parameters_unchanged']
        item['parameter_scope'] = {k: {'parameters': v['parameters']} for k, v in item['parameter_scope'].items()}
        cross = item['left_right_mean_visual_gradients']['unclipped']
        mean_norm = {}
        for name, value in cross.items():
            a, b, cosine = value['first_norm'], value['second_norm'], value['cosine']
            mean_norm[name] = math.sqrt(max((a*a + b*b + 2*a*b*(cosine or 0)) / 4, 0))
        total = mean_norm['twm_all']
        item['derived_mean_of_four_eval_visual_gradients'] = {
            'twm_norm': total, 'rotation_squared_norm_fraction': mean_norm['twm_rotation_head']**2 / total**2,
            'translation_squared_norm_fraction': mean_norm['twm_translation_head']**2 / total**2,
            'encoder_squared_norm_fraction': mean_norm['twm_encoder']**2 / total**2,
            'visual_div51_still_clipped': total / 51 > 1,
            'scope': 'mean of the four separate eval losses before clipping; not the actual training mixed batch, dropout or Adam update'}
        gradients[label] = item
    age = read('age_and_tail_summary.json')
    result = {'schema': 'handloop.joint_failure_deep_diagnosis.v1',
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'report_code_revision': args.code_revision,
        'report_script_sha256': sha(Path(__file__)), 'source_revisions': {'main_experiments': driver['source_revision'],
            'extended_gradients': grad_driver['source_revision'], 'camera': '11159dd78616daf2846aeedc25a8c773ca24b6ac'},
        'motion_paired': numeric, 'real_state_interventions': real, 'geometry_bias': geometry,
        'camera_depth': camera, 'gradient_probes': gradients, 'age_and_tail': age,
        'evidence': evidence, 'scope': 'Frozen checkpoints and diagnostic gradients only; no optimizer/update, test split or best checkpoint changes. Original official-HaWoR frontend, fixed camera, GT crop and fitted-MANO reference limits retained.'}
    output = args.output_dir / 'joint_failure_deep_diagnosis.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6))
    fig.subplots_adjust(left=.07, right=.98, top=.86, bottom=.14, wspace=.28, hspace=.5)
    colors = ['#2269a8', '#dc862d', '#35866a']
    labels = ['Pretrained', 'Joint best300', 'Joint last1000']
    ax = axes[0, 0]
    for index, label in enumerate(('pretrained', 'best300', 'last1000')):
        values = [motion['by_checkpoint'][label]['equal_source_by_hand'][hand]['noisy']['metrics']['output/wrist_mm']
                  for hand in ('left', 'right')]
        x = np.arange(2) + (index-1)*.24
        ax.bar(x, values, width=.23, color=colors[index], label=labels[index], zorder=3)
        for px, value in zip(x, values): ax.text(px, value+.06, f'{value:.2f}', ha='center', fontsize=9)
    ax.axhline(4.55361, color='#777', linestyle='--', linewidth=1)
    ax.set_xticks([0, 1], ['Left', 'Right']); ax.set_ylim(0, 5.4)
    ax.set_ylabel('Wrist error (mm)'); ax.set_title('Pure TWM: paired synthetic restoration')
    ax.legend(fontsize=8, loc='upper left'); ax.grid(axis='y', alpha=.2)
    ax = axes[0, 1]
    branches = ['baseline.raw', 'best.raw', 'best.round1_state', 'best.round2_candidate', 'best.final']
    values = [geometry['test']['branches'][b]['handedness']['left']['mpjpe21_mm'] for b in branches]
    ax.bar(range(5), values, color=['#7d8998', '#dc862d', '#dc862d', '#dc862d', '#a25e1c'], zorder=3)
    for i, v in enumerate(values): ax.text(i, v+2, f'{v:.1f}', ha='center', fontsize=9)
    ax.set_xticks(range(5), ['HaWoR', 'Joint raw', 'Pass 1', 'Visual 2', 'Final'])
    ax.set_ylim(0, 130); ax.set_ylabel('Absolute MPJPE21 (mm)'); ax.set_title('Left test: compensation and last-pass error')
    ax.grid(axis='y', alpha=.2)
    ax = axes[1, 0]
    with (base/'geometry_bias_test_per_frame.csv').open() as handle: rows = list(csv.DictReader(handle))
    for branch, color, label in [('baseline.raw', '#7d8998', 'HaWoR'), ('best.final', colors[1], 'Joint best300'), ('last.final', colors[2], 'Joint last1000')]:
        values = np.sort([float(r['wrist_mm']) for r in rows if r['branch'] == branch and r['handedness'] == 'left'])
        ax.plot(values, np.arange(1, len(values)+1)/len(values), color=color, label=label, linewidth=1.8)
    ax.set_xscale('log'); ax.set_xlim(1, 5000); ax.set_ylim(0, 1.02)
    ax.set_xlabel('Left wrist error (mm; logarithmic axis)'); ax.set_ylabel('Fraction of test states')
    ax.set_title('Left test: better median, catastrophic tail'); ax.legend(fontsize=8); ax.grid(alpha=.2)
    ax = axes[1, 1]
    categories = [('joint_saved_history', 'left'), ('joint_saved_history', 'right'),
                  ('joint_oracle_history', 'left'), ('joint_oracle_history', 'right')]
    for index, model in enumerate(('pretrain', 'best')):
        values = [real['test']['models'][model]['controls'][control][hand]['output']['wrist_mm'] for control, hand in categories]
        x = np.arange(4) + (index-.5)*.34
        ax.bar(x, values, width=.32, color=colors[index], label=labels[index], zorder=3)
        for px, value in zip(x, values): ax.text(px, value+1, f'{value:.1f}', ha='center', fontsize=9)
    ax.set_xticks(range(4), ['Saved H / L', 'Saved H / R', 'GT past / L', 'GT past / R'])
    ax.set_ylim(0, 86); ax.set_ylabel('Wrist error (mm)'); ax.set_title('Same saved joint raw/candidate; TWM swap')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=.2)
    fig.suptitle('Joint-training diagnosis: adaptation, retention and failures', fontsize=16, y=.98)
    fig.text(.5, .93, 'Frozen weights; equal-source motion validation; paired fixed-camera visual test subset', ha='center', fontsize=11)
    fig.text(.07, .065, 'GT past is an oracle diagnostic. State swaps are fixed-context interventions, not retrained system ablations.', fontsize=10)
    fig.text(.07, .035, 'GT crops and fitted MANO reference; all original test recordings retained; best selected before these tests.', fontsize=10)
    plot = args.output_dir/'joint_failure_deep_diagnosis.png'
    fig.savefig(plot, dpi=170, facecolor='white')
    print(json.dumps({'summary': str(output), 'summary_bytes': output.stat().st_size,
                      'plot': str(plot), 'plot_bytes': plot.stat().st_size, 'all_experiments_complete': True}), flush=True)


if __name__ == '__main__':
    main()
