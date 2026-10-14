#!/usr/bin/env python3
"""Run on the ds login host: plan/start/status persistent Pod downloads."""
import argparse
import json
from pathlib import Path
import subprocess

CODE = "/sciclone/home/jchen57/project/Hand_Reconstruction"
DATA = "/sciclone/data10/jchen57/Hand_Reconstruction/datasets"
PYTHON = "/sciclone/home/jchen57/.conda/envs/zhou/bin/python"
DATASETS = ("oakink2", "hot3d", "umetrack")
ALL_DATASETS = DATASETS + ("dexycb", "arctic")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "start", "status"))
    parser.add_argument("--pod", default="jchen57-debug-mpd5p")
    parser.add_argument("--namespace", default="jchen57")
    parser.add_argument("--code", default=CODE)
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--python", default=PYTHON)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--datasets", nargs="+", choices=ALL_DATASETS, default=list(DATASETS))
    parser.add_argument("--credentials", default=CODE + "/config/arctic_credentials.local.json")
    args = parser.parse_args()
    config = vars(args)
    config["datasets"] = tuple(dict.fromkeys(args.datasets))
    prefix = ["kubectl", "exec", "-i", args.pod, "-n", args.namespace, "--", args.python]
    if args.action == "plan":
        for dataset in config["datasets"]:
            runner = "download_comparison_data.py" if dataset in ("dexycb", "arctic") else "download_twm_data.py"
            subprocess.run(prefix + [str(Path(args.code) / "scripts" / runner),
                                     "plan", dataset, "--root", args.data], check=True)
        return
    # This code executes in the Pod, never on the login host.
    program = "CONFIG = " + repr(config) + "\n" + r'''
import fcntl, json, os, pathlib, subprocess
from datetime import datetime, timezone
root=pathlib.Path(CONFIG['data'])
for dataset in CONFIG['datasets']:
    directory=root/dataset
    manifest=json.loads((directory/'download_manifest.json').read_text())
    script=pathlib.Path(CONFIG['code'])/manifest.get('runner','scripts/download_twm_data.py')
    launch_path=directory/'download_process.json'
    launch=json.loads(launch_path.read_text()) if launch_path.exists() else {}
    pid=launch.get('pid')
    proc=pathlib.Path(f'/proc/{pid}/cmdline') if pid else None
    # PIDs are ephemeral; verify the command as well as the numeric ID.
    alive=bool(proc and proc.exists() and str(script).encode() in proc.read_bytes().split(b'\0')
               and dataset.encode() in proc.read_bytes().split(b'\0'))
    if CONFIG['action']=='start' and not alive:
        if not (directory/'download_manifest.json').exists():
            raise RuntimeError(f'Missing manifest for {dataset}; run plan first')
        with (directory/'.download.lock').open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        command=[CONFIG['python'], '-u', str(script), 'download', dataset,
                 '--root', CONFIG['data'], '--workers', str(CONFIG['workers'])]
        if dataset=='arctic':
            command+=['--credentials',CONFIG['credentials']]
        with (directory/'download.log').open('ab', buffering=0) as log:
            process=subprocess.Popen(command,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, cwd=CONFIG['code'], close_fds=True)
        launch={'pid':process.pid,'pod':CONFIG['pod'],'started_utc':datetime.now(timezone.utc).isoformat(),
                'workers':CONFIG['workers'],'script':str(script)}
        temporary=launch_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(launch,indent=2)+'\n')
        temporary.replace(launch_path)
        alive=True
    status_path=directory/'download_status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {}
    print(json.dumps({'dataset':dataset,'process_alive':alive,'pid':launch.get('pid'),
                      'path':str(directory),'scope':manifest['scope'],
                      'selected_files':len(manifest['files']),'status':status}), flush=True)
'''
    subprocess.run(prefix + ["-"], input=program, text=True, check=True)


if __name__ == "__main__":
    main()
