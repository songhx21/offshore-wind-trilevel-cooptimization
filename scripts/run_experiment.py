"""Select one archived workflow; training requires the explicit --run flag."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--case', choices=['25T', '50T'], required=True)
parser.add_argument('--configuration', choices=['AWS', 'FAE', 'RUC', 'RNT'], required=True)
parser.add_argument('--workflow', choices=['BASE', 'SEQ', 'CO', 'TCO'], required=True)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--run', action='store_true', help='Actually launch the full optimization')
args = parser.parse_args()
stem = {'AWS':'v7', 'FAE':'v8-fatigue', 'RUC':'v9-robust', 'RNT':'v10-reliability'}[args.configuration]
suffix = {'BASE':'', 'SEQ':'-reopt', 'CO':'-coopt', 'TCO':'-coopt-optimized'}[args.workflow]
name = f'GNN-PPO-TriLevelYaw-{stem}{suffix}.py'
directory = ROOT / 'experiments' / args.case
assert (directory / name).is_file()
print(f'Case={args.case} configuration={args.configuration} workflow={args.workflow} seed={args.seed}')
print(f'Working directory: {directory}')
print(f'Command: {sys.executable} -u {name}')
if args.run:
    environment = os.environ.copy()
    environment['GNN_PPO_SEED'] = str(args.seed)
    environment['PYTHONHASHSEED'] = str(args.seed)
    environment.pop('GNN_PPO_RUN_NAME', None)
    sys.exit(subprocess.call([sys.executable, '-u', name], cwd=directory, env=environment))
print('Preview only. Add --run to start training. Existing archived data are not modified.')
