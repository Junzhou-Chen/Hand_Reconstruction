#!/usr/bin/env python3
"""Prepare official assets and isolated extras on the REMOTE compute Pod.

The official source is pinned and verified again by FrozenHaWoRBridge. Neither
Torch nor an existing training runtime is upgraded. Licensed MANO assets must
already be available; this command does not download them or handle credentials.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request
import uuid


ASSETS = {
    'source.tar.gz': (
        'https://codeload.github.com/ThunderVVV/HaWoR/tar.gz/66c7d4108d58a716deccd192cb7645170cdc7bd7',
        'c44ebbe8d2999be2a1e27036f52e325658e3229d68703bbcdcb63bcd9708d04b'),
    'weights/hawor.ckpt': (
        'https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt',
        '4d1cc43853c190d6f2c10d9b6295c73109f0faf9ef41ac817a2b31d94b4823f2'),
    'weights/model_config.yaml': (
        'https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/model_config.yaml',
        'edfe12dc14ce371d698da722b59acfed5b4a38a7f8f5116cbc1fce459a07dd2d'),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--assets', action='store_true')
    parser.add_argument('--install-extras', action='store_true')
    args = parser.parse_args()
    if not args.assets and not args.install_extras:
        parser.error('Choose --assets and/or --install-extras')
    asset_root = args.data_root / 'model_assets/hawor_v1'
    extra_root = args.data_root / 'runtime/hawor_extra'
    if args.assets:
        receipts = []
        for relative, (url, expected) in ASSETS.items():
            path = asset_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                temporary = path.with_suffix(path.suffix + '.part')
                with urllib.request.urlopen(url, timeout=120) as source, temporary.open('wb') as out:
                    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                        out.write(chunk)
                if sha256(temporary) != expected:
                    raise RuntimeError(f'Official asset hash changed: {relative}; inspect before updating pins')
                temporary.replace(path)
            if sha256(path) != expected:
                raise RuntimeError(f'Existing asset failed its pinned checksum: {relative}')
            receipts.append({'name': relative, 'url': url, 'sha256': expected, 'bytes': path.stat().st_size})
        if not (asset_root / 'source').exists():
            with tarfile.open(asset_root / 'source.tar.gz') as archive:
                folder = archive.getnames()[0].split('/')[0]
                archive.extractall(asset_root, filter='data')
            (asset_root / folder).rename(asset_root / 'source')
        (asset_root / 'download_receipt.json').write_text(json.dumps(receipts, indent=2) + '\n')
    if args.install_extras:
        import torch
        if not torch.__version__.startswith('2.10.') or torch.version.cuda != '13.0':
            raise RuntimeError('This extras recipe is for the established Torch 2.10/cu130 runtime')
        target = extra_root / 'site-packages'
        # Readers must never see pip's partially copied package directories.
        staging = extra_root / ('site-packages.build-' + uuid.uuid4().hex[:12])
        staging.mkdir(parents=True, exist_ok=False)
        requirements = Path(__file__).resolve().parents[1] / 'config/hawor_extra_requirements.txt'
        base = [sys.executable, '-m', 'pip', 'install', '--target', str(staging), '--no-deps', '--no-compile']
        subprocess.run(base + ['-r', str(requirements)], check=True)
        subprocess.run(base + ['torchvision==0.25.0', '--index-url', 'https://download.pytorch.org/whl/cu130'], check=True)
        env = os.environ.copy()
        env['PYTHONPATH'] = str(staging)
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        subprocess.run([sys.executable, '-c',
            'import scipy.linalg, skimage.transform, torchvision, pytorch_lightning, timm, cv2, yacs; '
            'print("isolated extras imports passed")'], check=True, env=env)
        if target.exists():
            target.rename(extra_root / ('site-packages.previous-' + uuid.uuid4().hex[:12]))
        staging.rename(target)
        (extra_root / 'install_receipt.json').write_text(json.dumps({
            'torch_untouched': torch.__version__, 'cuda': torch.version.cuda,
            'requirements_sha256': sha256(requirements), 'target': str(target),
            'torchvision': '0.25.0+cu130', 'policy': 'isolated target; --no-deps'}, indent=2) + '\n')
    print(json.dumps({'assets': str(asset_root), 'extra_pythonpath': str(extra_root / 'site-packages')}))


if __name__ == '__main__':
    main()
