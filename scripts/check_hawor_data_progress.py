#!/usr/bin/env python3
"""Read download and preprocessing progress inside the existing ds Pod.

Run from the ds login host. This never reads private credentials, restarts
workers, loads models, processes dataset samples or starts training.
"""
from __future__ import annotations

import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pod', default='jchen57-fineturn-tl4zr')
    parser.add_argument('--namespace', default='jchen57')
    parser.add_argument('--root', default='/sciclone/data10/jchen57/Hand_Reconstruction')
    args = parser.parse_args()
    program = 'ROOT = ' + repr(args.root) + '\n' + r'''
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

root = Path(ROOT)

def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}

def alive(receipt):
    pid = receipt.get('pid')
    proc = Path(f'/proc/{pid}/cmdline')
    if not pid or not proc.is_file():
        return False
    command = receipt.get('command') or []
    script = receipt.get('script')
    if not script:
        script = next((value for value in command if value.endswith('.py')), None)
    try:
        return bool(script and str(script).encode() in proc.read_bytes().split(b'\0'))
    except FileNotFoundError:
        return False

print('查询时间：' + datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S %Z'))
for directory, label in [('dexycb', 'DexYCB'), ('arctic', 'ARCTIC'), ('ho3d_v2', 'HO3D v2')]:
    location = root / 'datasets' / directory
    status = read(location / 'download_status.json')
    phase = status.get('phase', '尚无状态文件')
    size = status.get('present_bytes')
    text = f'{label}: phase={phase}'
    if isinstance(size, (int, float)):
        text += f'；已写入 {size / 1e9:.2f} GB'
    counts = status.get('counts')
    if counts:
        total = len(read(location / 'download_manifest.json').get('files', []))
        text += f'；verified={counts.get("verified", "?")}/{total or "?"}；failed={counts.get("failed", "?")}'
    if directory == 'ho3d_v2' and phase == 'complete':
        text += f'；ZIP CRC通过={status.get("zip_crc_passed")}；SHA256已记录={bool(status.get("sha256"))}'
    if phase not in ('complete', 'failed', 'authentication_failed', 'waiting_credentials'):
        text += '；进程=' + ('运行中' if alive(read(location / 'download_process.json')) else '已退出，请检查')
    print(text)
    if phase in ('failed', 'authentication_failed'):
        message = status.get('message') or status.get('errors')
        if message:
            print('  原因：' + str(message)[:500])

prepared = root / 'prepared/visual_hot3d_aria_all_downloaded_v4_20261016'
manifest = read(prepared / 'source_manifest.json')
if manifest.get('phase') == 'complete':
    print('HOT3D RGB预处理: complete；有效手状态=' + str(manifest.get('split_frames')))
else:
    receipt = read(root / 'evaluations/hawor_training_source_alignment_20261016/hot3d_all_clips_process.json')
    log = Path(receipt['log']) if receipt.get('log') else None
    progress = None
    if log and log.is_file():
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size - 16384))
            lines = stream.read().decode(errors='replace').splitlines()
        for line in reversed(lines):
            try:
                item = json.loads(line)
                if 'complete_clips' in item:
                    progress = item
                    break
            except ValueError:
                pass
    state = '运行中' if alive(receipt) else '已退出，未生成完整manifest，请检查'
    count = f'{progress["complete_clips"]}/{progress["total_clips"]} clips' if progress else '暂无完成计数'
    print(f'HOT3D RGB预处理: {count}；{state}')
print('complete表示下载及校验完成；checking表示仍在校验。下载完成不等于四库训练接口已验收。')
'''
    command = ['kubectl', 'exec', '-i', args.pod, '-n', args.namespace, '--',
               '/sciclone/home/jchen57/.conda/envs/zhou/bin/python', '-']
    subprocess.run(command, input=program, text=True, check=True)


if __name__ == '__main__':
    main()
