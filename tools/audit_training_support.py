"""CPU diagnostic: actual initialization updates and source-word support.

Potential word supports come from the exact registered training vocabulary,
not actual sampled SGD event counts. Jaccard is diagnostic, never convergence
evidence. No new weights, synthetic vectors, downloads or GPU operations.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time
import zipfile

import numpy as np

from .benchmark_large_structure import distribution, row_metrics, sample_pairs, sample_siblings, write_json
from .benchmark_quantized_capacity import software_environment
from .benchmark_memory_capacity import memory
from .prepare_public_benchmark import file_sha256


def jaccard(a,b):
    a,b=set(a),set(b)
    return len(a & b)/len(a | b) if a or b else 1.


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-dir',type=Path,required=True)
    p.add_argument('--corpus-zip',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--sample-per-order',type=int,default=256)
    p.add_argument('--seed',type=int,default=42)
    cfg=p.parse_args(); started=time.perf_counter()
    training=json.loads((cfg.training_dir/'training.json').read_text())
    for name,field in [('trained.f32','trained_sha256'),('initial.f32','initial_sha256'),
                       ('trained-source-ids.u64','source_row_ids_sha256'),('keys.json','keys_sha256')]:
        if file_sha256(cfg.training_dir/name)!=training[field]:
            raise ValueError('training provenance mismatch: '+name)
    if file_sha256(cfg.corpus_zip)!=training['zip_sha256']:
        raise ValueError('corpus archive mismatch')
    keys=json.loads((cfg.training_dir/'keys.json').read_text())
    ids=np.fromfile(cfg.training_dir/'trained-source-ids.u64',dtype='<u8')
    shape=(len(keys),training['dim']); orders=np.asarray([len(k) for k in keys])
    rows=np.memmap(cfg.training_dir/'trained.f32',mode='r',dtype='<f4',shape=shape)
    initial=np.memmap(cfg.training_dir/'initial.f32',mode='r',dtype='<f4',shape=(training['registered_ngram_rows'],shape[1]))
    if len(ids)!=len(keys) or np.any(ids>=len(initial)) or len(np.unique(ids))!=len(ids):
        raise ValueError('invalid original training row mapping')
    update=np.empty(len(keys));cosine=np.empty(len(keys));changed=np.zeros(len(keys),dtype=bool)
    for start in range(0,len(keys),4096):
        stop=min(start+4096,len(keys));old=initial[ids[start:stop]];new=rows[start:stop]
        changed[start:stop]=np.any(old.view('u4')!=new.view('u4'),axis=1)
        metric=row_metrics(old,new)
        update[start:stop]=metric['relative_l2'];cosine[start:stop]=metric['cosine']
    if not changed.all():
        raise ValueError('export contains unchanged initialization rows')
    initial._mmap.close(); rows._mmap.close()
    pairs=sample_pairs(keys,cfg.sample_per_order,cfg.seed)+sample_siblings(keys,cfg.sample_per_order,cfg.seed)
    tracked={i for entry in pairs for edge in entry['pairs'] for i in edge}
    supports={i:[] for i in tracked}
    count=np.zeros(len(keys),dtype=np.int64);mass=np.zeros(len(keys),dtype=np.int64)
    with zipfile.ZipFile(cfg.corpus_zip) as archive:
        raw=archive.read('enwik8')
    if hashlib.sha256(raw).hexdigest()!=training['corpus_raw_sha256']:
        raise ValueError('corpus raw hash mismatch')
    text=re.sub(r'<[^>]*>',' ',raw[:int(len(raw)*.9)].decode('utf-8',errors='replace'))
    counts=Counter(w for w in re.findall(r"[a-z]+(?:'[a-z]+)?",text.lower()) if 2<=len(w)<=64)
    words=sorted(w for w,c in counts.items() if c>=training['settings']['min_count'])
    if len(words)!=training['vocabulary']:
        raise ValueError('source vocabulary mismatch')
    lookup={g:i for i,g in enumerate(keys)}
    for wid,word in enumerate(words):
        bounded='<'+word+'>'
        bag=Counter(bounded[i:i+n] for i in range(len(bounded)) for n in range(2,training['settings']['order']+1) if i+n<=len(bounded))
        for g,multiplicity in bag.items():
            row=lookup.get(g)
            if row is None:
                continue
            count[row]+=1;mass[row]+=counts[word]*multiplicity
            if row in supports:
                supports[row].append(wid)
    strata=[]
    rows=np.memmap(cfg.training_dir/'trained.f32',mode='r',dtype='<f4',shape=shape)
    for entry in pairs:
        values=np.asarray([jaccard(supports[i],supports[j]) for i,j in entry['pairs']])
        pair_ids=np.asarray(entry['pairs'],dtype=np.int64)
        metric=row_metrics(rows[pair_ids[:,0]],rows[pair_ids[:,1]])
        identical=values==1
        strongly_updated=(update[pair_ids[:,0]]>5)&(update[pair_ids[:,1]]>5)
        joint=(metric['cosine']>=.99)&(metric['norm_ratio']>=.99)&metric['nonzero']
        strata.append(dict(order=entry['order'],target_order=entry['target_order'],direction=entry['direction'],
            pairs=len(values),source_word_support_jaccard=distribution(values),
            identical_support_fraction=float(np.mean(values==1)),
            cosine_when_identical_support=distribution(metric['cosine'][identical]),
            norm_ratio_when_identical_support=distribution(metric['norm_ratio'][identical]),
            joint099_fraction_when_identical_support=float(joint[identical].mean()) if identical.any() else None,
            cosine_when_both_updates_exceed_5x_initial=distribution(metric['cosine'][strongly_updated]),
            pair_ids_sha256=hashlib.sha256(np.asarray(entry['pairs'],dtype='<u8').tobytes()).hexdigest()))
    rows._mmap.close()
    result=dict(status='passed',**software_environment(),source_sha256=training['trained_sha256'],
        initial_sha256=training['initial_sha256'],source_ids_sha256=training['source_row_ids_sha256'],
        corpus_sha256=training['corpus_raw_sha256'],all_exported_rows_changed=True,
        sample_per_order=cfg.sample_per_order,seed=cfg.seed,strata=strata,by_order={},
        limitations=['One finite rare-center-coverage SGNS checkpoint; update magnitude does not establish convergence.',
            'Source-word support is potential co-training membership in the registered vocabulary, not the actual sampled SGD update count.',
            'Character-length strata from one order-12 table; nearest siblings are biased diagnostics, not original-planner candidates.',
            'Cosine to actual initialization is a training diagnostic, not randomized vector compression evidence.'])
    for order in np.unique(orders):
        mask=orders==order
        groups={}
        for name,lo,hi in [('one-word',1,1),('two-to-four',2,4),('five-to-twenty',5,20),('twenty-one-plus',21,len(words))]:
            selected=mask & (count>=lo) & (count<=hi)
            groups[name]=dict(rows=int(selected.sum()),training_update_relative_to_initial=distribution(update[selected]))
        result['by_order'][str(int(order))]=dict(rows=int(mask.sum()),
            distinct_source_words=distribution(count[mask]),potential_corpus_occurrence_mass=distribution(mass[mask]),
            one_source_word_fraction=float(np.mean(count[mask]==1)),
            training_update_relative_to_initial=distribution(update[mask]),cosine_to_initial=distribution(cosine[mask]),
            update_below_10_percent_fraction=float(np.mean(update[mask]<.1)),
            update_below_100_percent_fraction=float(np.mean(update[mask]<1)),by_support=groups)
    result.update(wall_seconds=time.perf_counter()-started,memory_after=memory())
    write_json(cfg.report,result)


if __name__=='__main__':
    main()
