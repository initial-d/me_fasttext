"""Join independent encoding runs after full builds and row verification.

The combined report is builds-completed until missing fresh query workers run.
Every input report/hash/status is retained; completed encoding slices of a
stopped multi-encoding controller are explicitly labeled, never called full runs.
"""
import argparse
import copy
import json
from pathlib import Path

from .benchmark_large_structure import write_json
from .prepare_public_benchmark import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--part',action='append',required=True,help='encoding=/absolute/report.json')
    p.add_argument('--output',type=Path,required=True)
    cfg=p.parse_args()
    combined=None; index_hashes=None
    for part in cfg.part:
        encoding,path=part.split('=',1);path=Path(path)
        report=json.loads(path.read_text());info=report['provenance']
        if combined is None:
            combined=dict(status='builds-completed',provenance=copy.deepcopy(info),workers=[],variants=[],parts=[])
            combined['provenance']['settings']['encodings']=[]
        for field in ('source_sha256','keys_sha256','shape','trace_sha256','native_library_sha256','structural_library_sha256'):
            if info[field]!=combined['provenance'][field]:
                raise ValueError('mismatched immutable input: '+field)
        for field in ('thresholds','norm_ratio','trace_rows','batch_rows','rounds','repeat','max_block_bytes','seed'):
            if info['settings'][field]!=combined['provenance']['settings'][field]:
                raise ValueError('mismatched protocol: '+field)
        if encoding in combined['provenance']['settings']['encodings']:
            raise ValueError('duplicate encoding part')
        variants=[v for v in report['variants'] if v['encoding']==encoding]
        expected={encoding+'-'+name for name in ('raw','identity','layout','encoded-dedup')}
        expected.update(encoding+'-joint-'+str(t) for t in info['settings']['thresholds'])
        if {v['name'] for v in variants}!=expected or len(variants)!=len(expected):
            raise ValueError('encoding must contain every prescribed complete variant')
        for v in variants:
            check=v['verification']
            if (check['status']!='passed' or check['all_rows_verified']!=info['shape'][0]
                or not check['all_encoded_and_decoded_original_ids_byte_exact_to_representative']):
                raise ValueError('full encoding verification missing')
        selected=[w for w in report['workers'] if w['name'].startswith(tuple(
            prefix+encoding+'-' for prefix in ('plan-','build-','verify-','query-')))]
        if any(w['returncode'] or w['result']['status']!='passed' for w in selected):
            raise ValueError('failed selected worker')
        hashes={name:file_sha256(path.parent/name) for name in ('keys.das','keys.flat')}
        if index_hashes is not None and hashes!=index_hashes:
            raise ValueError('serialized key indices differ')
        index_hashes=hashes
        combined['index_bytes']={name:(path.parent/name).stat().st_size for name in hashes}
        combined['index_sha256']=hashes
        if not combined['workers']:
            combined['workers'].extend(w for w in report['workers'] if w['phase']=='audit')
        combined['workers'].extend(selected)
        combined['variants'].extend(variants)
        combined['provenance']['settings']['encodings'].append(encoding)
        combined['parts'].append(dict(encoding=encoding,report=str(path),report_sha256=file_sha256(path),
            original_controller_status=report['status'],source_commit=info['source_commit'],
            selection='completed and fully verified encoding slice; all original statuses retained'))
    write_json(cfg.output,combined)


if __name__=='__main__':
    main()
