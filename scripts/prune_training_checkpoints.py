#!/usr/bin/env python3
"""Plan or apply best-only cleanup of explicitly scoped remote experiment weights.

Dry-run (the default) writes a reviewable JSON plan, including CPU numerical
audits of every retained best. Apply requires that plan and rechecks inventory,
file identity, and all retained best before unlinking any obsolete weight.
Metadata, logs, scripts, and explicitly excluded files/directories are preserved.
"""

import argparse
import gc
import hashlib
import json
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
import re
import stat
import sys


PROJECT = Path('/sciclone/data10/jchen57/Hand_Reconstruction')
REMOTE_GUARD = Path('/sciclone/home/jchen57')
ROOT_NAMES = {'runs', 'audits', 'evaluations'}
WEIGHT_ENDING = re.compile(r'\.(?:pt|pth|ckpt)(?:\.tmp)?$', re.IGNORECASE)
BEST_NAME = re.compile(r'^best(?:[_.-].*)?\.(?:pt|pth|ckpt)$', re.IGNORECASE)
TRAINING_METADATA = {
    'run_metadata.json', 'training_metadata.json', 'metadata.json',
    'status.json', 'metrics.jsonl', 'training_state.json',
}
SCHEMA = 'handloop.best_only_checkpoint_cleanup.v1'


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def under(path, parent):
    return path == parent or parent in path.parents


def file_stat(path):
    value = path.lstat()
    return {
        'bytes': value.st_size, 'mtime_ns': value.st_mtime_ns,
        'ctime_ns': value.st_ctime_ns, 'inode': value.st_ino,
        'device': value.st_dev, 'mode': value.st_mode,
    }


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def normalize_scopes(args, parser):
    project = PROJECT.resolve(strict=True)
    roots = sorted({path.resolve(strict=True) for path in args.root})
    for path in roots:
        if path.name not in ROOT_NAMES or not under(path, project) or not path.is_dir():
            parser.error(f'Unsafe scope {path}: require a project runs/audits/evaluations directory')
    for index, left in enumerate(roots):
        if any(under(right, left) for right in roots[index + 1:]):
            parser.error('Overlapping --root scopes are not supported')

    def scoped_paths(values, label):
        result = sorted({path.resolve(strict=False) for path in values})
        for path in result:
            if not any(under(path, root) for root in roots):
                parser.error(f'{label} lies outside supplied roots: {path}')
            if path in roots:
                parser.error(f'{label} must name a child file/directory: {path}')
        return result

    excludes = scoped_paths(args.exclude, '--exclude')
    allowed = scoped_paths(args.allow_no_best, '--allow-no-best')
    if any(not path.is_dir() for path in allowed):
        parser.error('--allow-no-best must identify an existing reviewed directory')
    return roots, excludes, allowed


def inventory(roots, excludes, allowed):
    candidates, other_files, symlinks, excluded = [], [], [], set()
    markers = {}

    def is_excluded(path):
        for skip in excludes:
            if under(path, skip):
                excluded.add(str(skip))
                return True
        return False

    for root in roots:
        for directory, dirs, filenames in os.walk(root, followlinks=False):
            directory = Path(directory)
            kept_dirs = []
            for name in sorted(dirs):
                path = directory / name
                if is_excluded(path):
                    continue
                if path.is_symlink():
                    symlinks.append({'path': str(path), 'kind': 'directory', 'preserved': True})
                    continue
                kept_dirs.append(name)
            dirs[:] = kept_dirs
            for name in sorted(filenames):
                path = directory / name
                if is_excluded(path):
                    continue
                relative = path.relative_to(root)
                unit = root / relative.parts[0] if len(relative.parts) > 1 else root
                record = {'path': str(path), 'root': str(root), 'unit': str(unit),
                          'stat': file_stat(path)}
                if WEIGHT_ENDING.search(name):
                    candidates.append(record)
                else:
                    other_files.append(record)
                    if name in TRAINING_METADATA:
                        markers.setdefault(str(unit), []).append(str(path))

    by_directory = {}
    for record in candidates:
        by_directory.setdefault(str(Path(record['path']).parent), []).append(record)
    for records in by_directory.values():
        canonical = any(Path(item['path']).name == 'best.pt' for item in records)
        for record in records:
            name = Path(record['path']).name
            is_best = name == 'best.pt' if canonical else bool(BEST_NAME.match(name))
            record['action'] = 'keep_best' if is_best else 'remove_obsolete_weight'
            record['reason'] = ('canonical best.pt' if canonical else 'best alias (no canonical best.pt)') if is_best else 'best-only retention'

    blockers = []
    for record in candidates:
        if not stat.S_ISREG(record['stat']['mode']):
            blockers.append(f'Non-regular checkpoint candidate requires review: {record["path"]}')
    by_unit = {}
    for record in candidates:
        by_unit.setdefault(record['unit'], []).append(record)
    allowed_strings = {str(path) for path in allowed}
    for unit, records in sorted(by_unit.items()):
        has_best = any(item['action'] == 'keep_best' for item in records)
        if not has_best and unit in markers and unit not in allowed_strings:
            blockers.append(f'Training metadata but no retained best in {unit}; review and exclude, or explicitly --allow-no-best this directory')
    return {
        'candidates': sorted(candidates, key=lambda value: value['path']),
        'preserved_other_files': sorted(other_files, key=lambda value: value['path']),
        'preserved_directory_symlinks': symlinks,
        'excluded_paths': sorted(excluded),
        'training_metadata_by_unit': markers,
        'reviewed_no_best_units': sorted(allowed_strings),
        'blockers': blockers,
    }


def audit_best(record):
    import torch

    path = Path(record['path'])
    before_stat = file_stat(path)
    if before_stat != record['stat']:
        raise RuntimeError(f'Best changed since inventory: {path}')
    before_hash = sha256(path)
    # These are trusted checkpoints generated by this research project, not
    # arbitrary downloaded pickle files. CPU mapping prevents GPU allocation.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    counts = {'floating_tensors': 0, 'complex_tensors': 0, 'all_tensors': 0,
              'floating_elements': 0}
    seen = set()

    def visit(value, location='checkpoint'):
        if torch.is_tensor(value):
            if id(value) in seen:
                return
            seen.add(id(value))
            counts['all_tensors'] += 1
            if value.is_floating_point() or value.is_complex():
                key = 'complex_tensors' if value.is_complex() else 'floating_tensors'
                counts[key] += 1
                counts['floating_elements'] += value.numel()
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError(f'Nonfinite retained best tensor at {path}:{location}')
        elif isinstance(value, Mapping):
            for key, child in value.items():
                visit(child, f'{location}.{key}')
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f'{location}[{index}]')

    visit(payload)
    if not counts['floating_tensors'] and not counts['complex_tensors']:
        raise ValueError(f'No floating model tensors found in retained best: {path}')
    checkpoint_step = payload.get('step') if isinstance(payload, Mapping) else None
    if torch.is_tensor(checkpoint_step):
        checkpoint_step = checkpoint_step.item() if checkpoint_step.numel() == 1 else str(checkpoint_step)
    result = {'path': str(path), **counts, 'all_floating_tensors_finite': True,
              'checkpoint_step': checkpoint_step,
              'sha256_before': before_hash, 'sha256_after': sha256(path)}
    del payload
    gc.collect()
    if before_stat != file_stat(path) or result['sha256_before'] != result['sha256_after']:
        raise RuntimeError(f'Best changed during CPU audit: {path}')
    return result


def checkpoint_signature(value):
    return [{key: record[key] for key in ('path', 'root', 'unit', 'stat', 'action')}
            for record in value['candidates']]


def summarize(value):
    return {
        'keep_best_files': sum(item['action'] == 'keep_best' for item in value['candidates']),
        'keep_best_bytes': sum(item['stat']['bytes'] for item in value['candidates'] if item['action'] == 'keep_best'),
        'remove_files': sum(item['action'] == 'remove_obsolete_weight' for item in value['candidates']),
        'remove_bytes': sum(item['stat']['bytes'] for item in value['candidates'] if item['action'] == 'remove_obsolete_weight'),
        'preserved_other_files': len(value['preserved_other_files']),
        'preserved_other_bytes': sum(item['stat']['bytes'] for item in value['preserved_other_files']),
    }


def write_json(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, action='append', required=True,
                        help='Repeatable project runs/audits/evaluations root')
    parser.add_argument('--exclude', type=Path, action='append', default=[],
                        help='Preserve an entire child directory or exact file; repeatable')
    parser.add_argument('--allow-no-best', type=Path, action='append', default=[],
                        help='Explicitly reviewed smoke/output directory whose weights may all be removed')
    parser.add_argument('--output', type=Path, required=True, help='JSON plan or application receipt')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--plan', type=Path, help='Previously reviewed dry-run JSON plan; required with --apply')
    args = parser.parse_args()
    if not REMOTE_GUARD.is_dir():
        parser.error('Run only inside the authorized SciClone remote environment')
    if args.apply != (args.plan is not None):
        parser.error('Use --apply and --plan together; omit both for a dry-run')
    roots, excludes, allowed = normalize_scopes(args, parser)
    output = args.output.resolve(strict=False)
    if args.plan and output == args.plan.resolve(strict=True):
        parser.error('Application --output must differ from the reviewed --plan')
    if WEIGHT_ENDING.search(output.name):
        parser.error('--output must be JSON, never a checkpoint path')
    scope = {'roots': [str(path) for path in roots],
             'excludes': [str(path) for path in excludes],
             'allow_no_best': [str(path) for path in allowed]}
    result = {'schema': SCHEMA, 'created_at_utc': utc_now(),
              'mode': 'apply' if args.apply else 'dry_run', 'scope': scope,
              'policy': 'Keep best.pt per directory, or best aliases where no best.pt exists; preserve all non-checkpoint files and exclusions',
              'best_audits': [], 'removed': [], 'completed': False}
    exit_code = 0
    try:
        current = inventory(roots, excludes, allowed)
        result.update(current)
        result['summary'] = summarize(current)
        previous = None
        if args.plan:
            previous = json.loads(args.plan.read_text())
            if previous.get('schema') != SCHEMA or previous.get('mode') != 'dry_run' or not previous.get('completed'):
                raise ValueError('The supplied plan is not a completed dry-run of this script')
            if previous['scope'] != scope:
                raise ValueError('Apply roots/excludes/allow-no-best must exactly match the reviewed plan')
            if checkpoint_signature(previous) != checkpoint_signature(current):
                raise RuntimeError('Checkpoint inventory changed since plan; create and review a new dry-run')
        if current['blockers']:
            raise RuntimeError('Inventory has review blockers; no files deleted')
        for record in current['candidates']:
            if record['action'] == 'keep_best':
                result['best_audits'].append(audit_best(record))
        if previous is not None:
            prior_hashes = {item['path']: item['sha256_after'] for item in previous['best_audits']}
            if any(prior_hashes.get(item['path']) != item['sha256_after'] for item in result['best_audits']):
                raise RuntimeError('Retained best contents differ from reviewed plan')
        if args.apply:
            fresh = inventory(roots, excludes, allowed)
            if fresh['blockers'] or checkpoint_signature(fresh) != checkpoint_signature(current):
                raise RuntimeError('Checkpoint inventory changed during best audit; no files deleted')
            # Save the completed safety checks before the first irreversible unlink.
            result['pre_delete_checks_completed_at_utc'] = utc_now()
            write_json(output, result)
            for record in current['candidates']:
                if record['action'] != 'remove_obsolete_weight':
                    continue
                path = Path(record['path'])
                if file_stat(path) != record['stat']:
                    raise RuntimeError(f'Obsolete weight changed before unlink; stopping: {path}')
                path.unlink()
                result['removed'].append({'path': str(path), 'bytes': record['stat']['bytes'],
                                          'original_stat': record['stat']})
                write_json(output, result)
            # Best tensors were audited above; verify their files still match after cleanup.
            for audit in result['best_audits']:
                if sha256(Path(audit['path'])) != audit['sha256_after']:
                    raise RuntimeError(f'Retained best changed after cleanup: {audit["path"]}')
            result['removed_bytes'] = sum(item['bytes'] for item in result['removed'])
        result['completed'] = True
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
        exit_code = 2
    result['removed_bytes'] = sum(item['bytes'] for item in result['removed'])
    result['finished_at_utc'] = utc_now()
    write_json(output, result)
    print(json.dumps({'output': str(output), 'mode': result['mode'],
                      'completed': result['completed'], 'summary': result.get('summary'),
                      'removed_files': len(result['removed']),
                      'removed_bytes': result.get('removed_bytes', 0),
                      'error': result.get('error')}, sort_keys=True), flush=True)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
