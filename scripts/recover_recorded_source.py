"""Recover a recorded script byte-for-byte without executing training."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[1]


def recover(run_id):
    rows=list(csv.DictReader((ROOT/'data/code_provenance.csv').open(encoding='utf-8')))
    row=next(r for r in rows if r['run_id']==run_id)
    case,version,workflow,_=run_id.split('_')
    stem={'V7':'v7','V8':'v8-fatigue','V9':'v9-robust','V10':'v10-reliability'}[version]
    suffix={'BASE':'','SEQ':'-reopt','CO':'-coopt','TCO':'-coopt-optimized'}[workflow]
    source=ROOT/f'experiments/{case}/GNN-PPO-TriLevelYaw-{stem}{suffix}.py'
    data=source.read_bytes()
    seed_matches=json.loads((ROOT/'data/seed_hash_recovery.json').read_text(encoding='utf-8'))
    for match in seed_matches:
        if match['run_id']==run_id:
            text=data.decode('utf-8-sig').replace('\r\n','\n')
            text=re.sub(r'(?m)^(DEFAULT_RANDOM_SEED\s*=\s*)\d+',lambda m:m[1]+str(match['default_seed']),text)
            ending='\r\n' if match['line_ending']=='CRLF' else '\n'
            data=(b'\xef\xbb\xbf' if match['bom'] else b'')+text.replace('\n',ending).encode('utf-8')
    if run_id=='50T_V10_TCO_seed42':
        data=(ROOT/'historical/50T_V10_TCO_seed42.py').read_bytes()
    expected=row['recorded_script_sha256']
    if not expected or hashlib.sha256(data).hexdigest()!=expected:
        raise ValueError(f'Exact recorded script is not recovered for {run_id}')
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    parser.add_argument('--output',type=Path,help='Optional destination for recovered script bytes')
    args=parser.parse_args()
    data=recover(args.run_id)
    print('Verified SHA-256:',hashlib.sha256(data).hexdigest())
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_bytes(data)
        print('Written:',args.output)


if __name__=='__main__':
    main()
