#!/usr/bin/env python3
"""Remote-only isolated Torch/CUDA/Mamba runtime with pinned official wheels.

Run using the existing zhou Python; this bootstraps and reexecutes inside a
fully isolated venv. It never installs into the shared zhou environment.
"""
from pathlib import Path
import sys, os, json, subprocess, hashlib, urllib.request, datetime, traceback
import argparse, venv
if not Path('/sciclone/home/jchen57').is_dir():
    raise RuntimeError('Create the experimental runtime on the remote Pod only')
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--runtime',type=Path,default=Path('/sciclone/data10/jchen57/Hand_Reconstruction/runtime/twm_mamba_torch210_v1'))
args=parser.parse_args()
root=args.runtime.resolve()
py=str(root/'bin/python')
if not Path(py).is_file():
    venv.EnvBuilder(with_pip=True,system_site_packages=False).create(root)
if Path(sys.prefix).resolve()!=root:
    os.execv(py,[py,str(Path(__file__).resolve()),'--runtime',str(root)])
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
receipt={'started_utc':started,'python':py,'base_environment_unchanged':True,'install_kind':'fully isolated venv, system_site_packages=False','wheels':[],'status':'installing'}
def save():
    (root/'runtime_receipt.json').write_text(json.dumps(receipt,indent=2))
def run(args):
    print('RUN',args,flush=True)
    subprocess.run(args,check=True,env={**os.environ,'PIP_DISABLE_PIP_VERSION_CHECK':'1','PIP_NO_CACHE_DIR':'1'})
save()
try:
    run([py,'-m','pip','install','--upgrade','pip','--report',str(root/'pip_bootstrap_report.json')])
    run([py,'-m','pip','install','torch==2.10.0','--index-url','https://download.pytorch.org/whl/cu130','--report',str(root/'torch_install_report.json')])
    import torch
    assert torch.__version__.startswith('2.10.0+cu130'),torch.__version__
    (root/'constraints.txt').write_text('torch==2.10.0+cu130\ntriton=='+__import__('importlib.metadata',fromlist=['version']).version('triton')+'\n')
    assets=[
        ('https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1%2Bcu13torch2.10cxx11abiTRUE-cp311-cp311-linux_x86_64.whl','mamba_ssm-2.3.2.post1+cu13torch2.10cxx11abiTRUE-cp311-cp311-linux_x86_64.whl','7bacd9decf359e122fdd6131c3591ba5d81fce86602732fa5e30426274b07983'),
        ('https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu13torch2.10cxx11abiTRUE-cp311-cp311-linux_x86_64.whl','causal_conv1d-1.7.0+cu13torch2.10cxx11abiTRUE-cp311-cp311-linux_x86_64.whl','46aee6009555c922bd06348d4845694f6e12ecbcd5d532ed526b6039e806fa16')]
    wheels=root/'wheels'; wheels.mkdir(exist_ok=True)
    files=[]
    for url,name,expected in assets:
        path=wheels/name; h=hashlib.sha256()
        print('DOWNLOAD',url,flush=True)
        with urllib.request.urlopen(url,timeout=120) as response,path.open('wb') as output:
            while True:
                chunk=response.read(8*1024*1024)
                if not chunk: break
                output.write(chunk);h.update(chunk)
        actual=h.hexdigest()
        if actual!=expected: raise RuntimeError('wheel SHA256 mismatch '+name)
        receipt['wheels'].append({'url':url,'path':str(path),'sha256':actual,'size':path.stat().st_size})
        files.append(str(path));save()
    run([py,'-m','pip','install','-c',str(root/'constraints.txt'),*files,'numpy','PyYAML','--report',str(root/'mamba_install_report.json')])
    run([py,'-m','pip','check'])
    (root/'pip_freeze.txt').write_text(subprocess.check_output([py,'-m','pip','freeze'],text=True))
    receipt['status']='installed';receipt['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
    print('INSTALL_SUCCESS',py,flush=True)
except BaseException as e:
    receipt['status']='failed';receipt['error']=repr(e);receipt['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save();traceback.print_exc();raise
