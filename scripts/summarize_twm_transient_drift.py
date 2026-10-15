#!/usr/bin/env python3
"""Remote CPU summary/figure of frozen transient-drift diagnostics."""
import argparse
import copy
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--figure', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/sciclone/home/jchen57').is_dir():
        parser.error('Run numerical summarization and rendering remotely')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    original = json.loads(args.input.read_text())
    summary = copy.deepcopy(original)
    threshold_audit = args.input.parent/'threshold_sensitivity.json'
    if threshold_audit.exists():
        summary['threshold_sensitivity'] = json.loads(threshold_audit.read_text())
    truth = np.asarray(summary.pop('truth_segment_positions_m'), dtype=np.float64)
    directions = np.asarray(summary.pop('offset_directions_world'), dtype=np.float64)
    for name, case in summary['scenarios'].items():
        for mode in ('observed_history', 'corrected_history'):
            if mode in case:
                position = np.asarray(case[mode].pop('positions'), dtype=np.float64)
                error = position-truth
                # GT-fitted constant alignment is a diagnostic decomposition only.
                aligned = error-error.mean(1, keepdims=True)
                case[mode]['oracle_constant_alignment'] = {
                    'unaligned_rmse_mm': float(np.sqrt(np.square(error).sum(-1).mean())*1000),
                    'aligned_rmse_mm': float(np.sqrt(np.square(aligned).sum(-1).mean())*1000),
                    'scope': 'One GT-fitted constant per trajectory over ten steps; unavailable without reference, not a deployed correction.'}
        steps = case['corrupt_steps_zero_based']
        if not steps or name.startswith('common_bias'):
            continue
        observed = truth.copy()
        for i, step in enumerate(steps):
            observed[:, step] += directions*(case['amplitude_mm']/1000)*((i+1)/len(steps) if name.startswith('ramp') else 1.)
        repaired = observed.copy()
        runs = []
        for step in steps:
            if runs and step == runs[-1][-1]+1:
                runs[-1].append(step)
            else:
                runs.append([step])
        for run in runs:
            left, right = min(run)-1, max(run)+1
            assert 0 <= left < right < 10
            for step in run:
                alpha = (step-left)/(right-left)
                repaired[:, step] = (1-alpha)*observed[:, left]+alpha*observed[:, right]
        error = np.linalg.norm(repaired-truth, axis=-1)*1000
        case['oracle_future_bridge'] = {'during_mean_mm': float(error[:, steps].mean()),
            'trace_mean_mm': error.mean(0).tolist(),
            'scope': 'Known corrupt interval boundaries plus later clean observations, linear in frame index. Noncausal oracle diagnostic, not a fair causal baseline or model result.'}
    summary['raw_positions_location'] = str(args.input)
    summary['summary_policy'] = 'Raw reference/predicted coordinates remain remote. Only aggregated errors and indices exported.'
    args.summary.write_text(json.dumps(summary, indent=2)+'\n')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    example = summary['noise_audit']['example']['error_norm_mm']
    ax.plot(range(1, 32), example, color='#D88324', linewidth=1.8, marker='o', markersize=3)
    ax.axvline(30.5, color='#758398', linestyle='--')
    ax.set_title('A  Training sampler: one medium-history example')
    ax.set_xlabel('30 historical frames, then current frame')
    ax.set_ylabel('Injected wrist error (mm)')
    ax.set_ylim(bottom=0)
    ax.text(.02, .04, '4,096 generated windows: no enclosed 3-4-frame\nruns above 6 mm in observed history',
        transform=ax.transAxes, fontsize=9, color='#46536B')
    handles = []
    for ax, key, title in zip(axes.flat[1:], ('block3_10mm', 'block4_10mm', 'block4_30mm'),
        ('B  Three consecutive errors, 10 mm', 'C  Four consecutive errors, 10 mm', 'D  Four consecutive errors, 30 mm')):
        case = summary['scenarios'][key]
        trace = case['observed_history']['trace_mean_error_mm']
        x = np.arange(1, 11)
        for label, values, color, style in (
            ('Observed input', trace['input'], '#323A45', '-'),
            ('Frozen TWM', trace['model'], '#2867B2', '-'),
            ('Causal 5-frame linear fit', trace['linear5'], '#D88324', '--'),
            ('Oracle bridge (uses future)', case['oracle_future_bridge']['trace_mean_mm'], '#23836B', ':')):
            line, = ax.plot(x, values, label=label, color=color, linestyle=style, linewidth=1.8, marker='o', markersize=3)
            if key == 'block3_10mm':
                handles.append(line)
        slots = case['corrupt_steps_zero_based']
        ax.axvspan(min(slots)+.5, max(slots)+1.5, color='#DCE4EF', alpha=.45, zorder=-1)
        ax.set_title(title)
        ax.set_xlabel('Step within the ten-frame segment')
        ax.set_ylabel('Mean wrist error (mm)')
        ax.set_xticks((1, 3, 5, 7, 10))
        ax.set_ylim(bottom=0)
    for ax in axes.flat:
        ax.grid(color='#E4E9EF', linewidth=.7)
        ax.set_axisbelow(True)
    fig.suptitle(f"Transient wrist drift: frozen checkpoint {summary['checkpoint_step']:,}", fontsize=16, fontweight='bold')
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .062), ncol=2, frameon=False)
    fig.text(.5, .037, '64 reference trajectories (16/source), synthetic wrist-only errors; causal TWM sees no future observations.',
        ha='center', fontsize=9, color='#46536B')
    fig.text(.5, .014, 'Oracle bridge knows bad-frame boundaries and uses later clean observations; it is a separate noncausal diagnostic.',
        ha='center', fontsize=8, color='#596579')
    fig.tight_layout(rect=(0, .14, 1, .95), h_pad=2., w_pad=2.)
    fig.savefig(args.figure, dpi=160, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'summary': str(args.summary), 'figure': str(args.figure),
        'motion': summary['reference_motion'],
        'oracle_bridges': {key: value.get('oracle_future_bridge', {}).get('during_mean_mm') for key, value in summary['scenarios'].items()},
        'constant_alignment': {key: value['observed_history']['oracle_constant_alignment'] for key, value in summary['scenarios'].items() if key in ('common_bias_10mm', 'block4_10mm')}}, indent=2))


if __name__ == '__main__':
    main()
