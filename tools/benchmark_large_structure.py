"""Audit real keyed rows and production DATrie/mark-compact at capacity scale.

CPU only; reuse a provenance-checked trained table. Length strata are slices of
one checkpoint, not models trained with different maximum orders. Pair sampling
never generates vector values. Full exports/verification use every source row.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shlex
import subprocess
import sys
import time
import zipfile

import numpy as np

from .benchmark_quantized_capacity import software_environment, read_chunks
from .benchmark_memory_capacity import memory, samples
from .prepare_public_benchmark import file_sha256
from .quantized_memory import QuantizedRows, encode_rows, decode_rows, pack_quantized_file, row_width
from .structural_native import StructuralPlan, StructuralIndex


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {'count': 0}
    return dict(count=len(values), mean=float(values.mean()), min=float(values.min()),
                p05=float(np.quantile(values, .05)), p50=float(np.median(values)),
                p95=float(np.quantile(values, .95)), max=float(values.max()))


def row_metrics(a, b):
    """Float64 direct replacement errors, including explicit zero-norm handling."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.sum(a*a, axis=1), np.sum(b*b, axis=1)
    delta = np.sum((a-b)**2, axis=1)
    nonzero = (na > 0) & (nb > 0)
    cosine = np.divide(np.sum(a*b, axis=1), np.sqrt(na*nb),
                       out=np.zeros(len(a)), where=nonzero)
    cosine[(na == 0) & (nb == 0)] = 1
    ratio = np.divide(np.sqrt(np.minimum(na, nb)), np.sqrt(np.maximum(na, nb)),
                      out=np.zeros(len(a)), where=np.maximum(na, nb) > 0)
    relative = np.sqrt(delta / np.maximum(na, np.finfo(np.float64).tiny))
    return dict(cosine=np.clip(cosine, -1, 1), norm_ratio=ratio, relative_l2=relative,
                square_error=delta, square_source=na, nonzero=nonzero)


def fidelity(a, b):
    m = row_metrics(a, b)
    return dict(rows=len(a), cosine=distribution(m['cosine']),
                norm_ratio=distribution(m['norm_ratio']), relative_l2=distribution(m['relative_l2']),
                relative_frobenius=float(np.sqrt(m['square_error'].sum() /
                    max(m['square_source'].sum(), np.finfo(np.float64).tiny))))


def related(a, b):
    return a != b and (a.startswith(b) or a.endswith(b) or b.startswith(a) or b.endswith(a))


def sample_pairs(keys, sample_per_order, seed):
    """Uniform source-row sampling per length, then all registered ancestors.

    Each (source length, ancestor length, direction) has at most one edge per
    source, so its eligible sampled edges are uniform within that stratum.
    Controls use the same source and target length/boundary markers, excluding
    actual prefix/suffix relations. Sources repeat across strata; not IID edges.
    """
    rng = np.random.default_rng(seed)
    lookup = {key: i for i, key in enumerate(keys)}
    pools = {}
    for i, key in enumerate(keys):
        pools.setdefault((len(key), key.startswith('<'), key.endswith('>')), []).append(i)
    orders = np.asarray([len(k) for k in keys])
    result = []
    for order in sorted(set(orders.tolist())):
        population = np.flatnonzero(orders == order)
        selected = rng.choice(population, min(sample_per_order, len(population)), replace=False)
        for target_order in range(2, order):
            for direction in ('prefix', 'suffix'):
                pairs, controls = [], []
                for source in selected:
                    key = keys[source]
                    target = key[:target_order] if direction == 'prefix' else key[-target_order:]
                    if target not in lookup:
                        continue
                    pairs.append((int(source), lookup[target]))
                    pool = pools[(target_order, target.startswith('<'), target.endswith('>'))]
                    other = None
                    for _ in range(32):
                        candidate = int(pool[rng.integers(len(pool))])
                        if not related(key, keys[candidate]):
                            other = candidate
                            break
                    if other is not None:
                        controls.append((int(source), other))
                if pairs:
                    result.append(dict(order=int(order), target_order=target_order, direction=direction,
                        source_population=len(population), sampled_sources=len(selected), pairs=pairs,
                        controls=controls))
    return result


def shared_affix(a, b, direction):
    if direction == 'suffix':
        a, b = a[::-1], b[::-1]
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))


def sample_siblings(keys, sample_per_order, seed):
    """Nearest lexical/reversed-lexical sibling for sampled same-length keys.

    This is an explicitly neighbor-biased diagnostic, not a uniform pair sample
    or a candidate set used by the original proper-ancestor production planner.
    """
    rng=np.random.default_rng(seed)
    orders=np.asarray([len(k) for k in keys])
    result=[]
    for order in np.unique(orders):
        population=np.flatnonzero(orders==order)
        selected=rng.choice(population,min(sample_per_order,len(population)),replace=False)
        pools={}
        for i in population:
            key=keys[i]
            pools.setdefault((key.startswith('<'),key.endswith('>')),[]).append(int(i))
        for direction in ('prefix','suffix'):
            ordered=sorted(population.tolist(),key=lambda i:keys[i] if direction=='prefix' else keys[i][::-1])
            positions={i:p for p,i in enumerate(ordered)}
            pairs=[];controls=[];lengths=[]
            for i in selected:
                pos=positions[i]
                neighbors=ordered[max(0,pos-1):pos]+ordered[pos+1:pos+2]
                if not neighbors:
                    continue
                other=max(neighbors,key=lambda j:shared_affix(keys[i],keys[j],direction))
                common=shared_affix(keys[i],keys[other],direction)
                if common<2:
                    continue
                pairs.append((int(i),other)); lengths.append(common)
                key=keys[i];pool=pools[(key.startswith('<'),key.endswith('>'))]
                for _ in range(32):
                    candidate=int(pool[rng.integers(len(pool))])
                    if candidate!=i and shared_affix(key,keys[candidate],direction)<2:
                        controls.append((int(i),candidate));break
            if pairs:
                result.append(dict(order=int(order),target_order=int(order),direction='sibling-'+direction,
                    source_population=len(population),sampled_sources=len(selected),pairs=pairs,controls=controls,
                    shared_affix_length=distribution(lengths),
                    sampling='nearest lexical neighbor, at least 2 shared characters; not production ancestor candidates'))
    return result


def pair_summary(rows, pairs, thresholds, norm_ratio):
    if not pairs:
        return dict(pairs=0)
    ids = np.asarray(pairs, dtype=np.int64)
    m = row_metrics(rows[ids[:, 0]], rows[ids[:, 1]])
    result = dict(pairs=len(ids), ids_sha256=hashlib.sha256(ids.astype('<u8').tobytes()).hexdigest(),
                  cosine=distribution(m['cosine']), norm_ratio=distribution(m['norm_ratio']),
                  replacement_relative_l2=distribution(m['relative_l2']), thresholds=[])
    for threshold in thresholds:
        c = (m['cosine'] >= threshold) & m['nonzero']
        r = (m['norm_ratio'] >= norm_ratio) & m['nonzero']
        result['thresholds'].append(dict(cosine=threshold, norm_ratio=norm_ratio,
            cosine_only_fraction=float(c.mean()), ratio_only_fraction=float(r.mean()),
            joint_fraction=float((c & r).mean()),
            ratio_only_cosine_failure_fraction=float((r & ~c).mean()),
            cosine_only_error=distribution(m['relative_l2'][c]),
            ratio_only_error=distribution(m['relative_l2'][r]),
            joint_error=distribution(m['relative_l2'][c & r])))
    return result


def source_info(cfg):
    training = json.loads((cfg.training_dir / 'training.json').read_text())
    keys = json.loads((cfg.training_dir / 'keys.json').read_text())
    shape = (training['ngram_rows'], training['dim'])
    if len(keys) != shape[0] or len(set(keys)) != len(keys):
        raise ValueError('one distinct registered key per source row required')
    if any(not isinstance(k, str) or not k or '\0' in k or not k.isascii() for k in keys):
        raise ValueError('this order audit requires the registered ASCII character n-grams')
    source = cfg.training_dir / 'trained.f32'
    if source.stat().st_size != shape[0] * shape[1] * 4:
        raise ValueError('source shape mismatch')
    return training, keys, shape, source


def prepare(cfg):
    training, keys, shape, source = source_info(cfg)
    for name, field in [('trained.f32', 'trained_sha256'), ('keys.json', 'keys_sha256'),
                        ('trained-source-ids.u64', 'source_row_ids_sha256'), ('queries.u64', 'query_ids_sha256')]:
        if file_sha256(cfg.training_dir / name) != training[field]:
            raise ValueError('training provenance hash mismatch: ' + name)
    queries = np.fromfile(cfg.training_dir / 'queries.u64', dtype='<u8')
    if len(queries) != training['query_rows'] or np.any(queries >= len(keys)):
        raise ValueError('invalid held-out trace')
    query_keys = json.loads((cfg.training_dir / 'query-keys.json').read_text())
    if [keys[i] for i in queries] != query_keys:
        raise ValueError('held-out key/ID alignment mismatch')
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    traces = {'heldout-lexical': queries[:cfg.trace_rows],
              'uniform-id-stress': np.random.default_rng(cfg.seed).integers(0, shape[0], cfg.trace_rows, dtype=np.uint64)}
    for name, ids in traces.items():
        ids.astype('<u8').tofile(cfg.output_dir / (name + '.u64'))
    # Only the n-gram mean component is exported, not word/output embeddings.
    bags = []
    if cfg.corpus_zip:
        if file_sha256(cfg.corpus_zip) != training['zip_sha256']:
            raise ValueError('corpus archive provenance mismatch')
        with zipfile.ZipFile(cfg.corpus_zip) as archive:
            raw = archive.read('enwik8')
        if hashlib.sha256(raw).hexdigest() != training['corpus_raw_sha256']:
            raise ValueError('corpus bytes mismatch')
        text = raw[int(len(raw)*.9):].decode('utf-8', errors='replace')
        words = re.findall(r"[a-z]+(?:'[a-z]+)?", re.sub(r'<[^>]*>', ' ', text).lower())
        lookup = {k:i for i,k in enumerate(keys)}
        missing = total = 0
        for word in words:
            if not 2 <= len(word) <= 64:
                continue
            bounded = '<' + word + '>'
            grams = [bounded[i:i+n] for i in range(len(bounded)) for n in range(2, training['settings']['order']+1)
                     if i+n <= len(bounded)]
            ids = [lookup[g] for g in grams if g in lookup]
            total += len(grams)
            missing += len(grams)-len(ids)
            if ids:
                bags.append(ids)
            if len(bags) >= cfg.bag_count:
                break
        write_json(cfg.output_dir / 'bags.json', bags)
        bag_info = dict(bags=len(bags), requested_ngram_occurrences=total, missing_ngram_occurrences=missing,
                        contract='mean of registered n-gram rows with multiplicity; absent rows omitted identically; not full word embeddings or task quality')
    else:
        bag_info = None
    info = dict(status='prepared', **software_environment(), settings={k:str(v) if isinstance(v, Path) else v
                for k,v in vars(cfg).items()}, training=training, shape=shape,
                training_report_sha256=file_sha256(cfg.training_dir/'training.json'),
                source_sha256=training['trained_sha256'], keys_sha256=training['keys_sha256'],
                keys_bytes=(cfg.training_dir/'keys.json').stat().st_size,
                bags=bag_info, trace_sha256={name:file_sha256(cfg.output_dir/(name+'.u64')) for name in traces},
                structural_library_sha256=file_sha256(cfg.structural_library),
                native_library_sha256=file_sha256(cfg.native_library),
                limitations=['One finite rare-center-coverage SGNS checkpoint; not converged training or Engram/LLM task quality.',
                    'Length strata use the same order-12 model, not separately trained order comparisons.',
                    'Samples are actual vectors; pair strata sample source IDs without replacement, with shared sources across strata.',
                    'Ratio-only and cosine-only are candidate audits, not production-planner/export modes.',
                    'Joint thresholds refer to quantized/dequantized rows; total distortion is also measured against original FP32.',
                    'Complete storage includes table, embedded maps/descriptors/scales, codec and structural serving metadata plus one common keys JSON; DATrie alternative reported separately.',
                    'Builder source, rounded copies and planning maps are offline files, excluded from serving size; builder RSS reported independently.',
                    'Fresh query workers on warmed shared filesystem; no disk-cold, exclusive CPU, cgroup pressure or whole-model throughput claim.',
                    'No GPU usage, retraining, downloads, energy or monetary cost measurement; publish aggregates/hashes only.'])
    write_json(cfg.output_dir/'provenance.json', info)


def audit(cfg):
    training, keys, shape, source = source_info(cfg)
    rows = np.memmap(source, mode='r', dtype='<f4', shape=shape)
    orders = np.asarray([len(k) for k in keys])
    norms = np.empty(shape[0], dtype=np.float64)
    for start in range(0, shape[0], cfg.chunk_rows):
        chunk = rows[start:start+cfg.chunk_rows].astype(np.float64)
        if not np.isfinite(chunk).all():
            raise ValueError('nonfinite source')
        norms[start:start+len(chunk)] = np.linalg.norm(chunk, axis=1)
    report = dict(status='running', **software_environment(), sample_per_order=cfg.sample_per_order,
        seed=cfg.seed, thresholds=cfg.thresholds, norm_ratio=cfg.norm_ratio,
        by_order={str(int(n)):dict(rows=int(np.sum(orders==n)), norm=distribution(norms[orders==n])) for n in np.unique(orders)},
        strata=[])
    for entry in sample_pairs(keys, cfg.sample_per_order, cfg.seed)+sample_siblings(keys,cfg.sample_per_order,cfg.seed):
        base = {k:v for k,v in entry.items() if k not in ('pairs', 'controls')}
        base['structural'] = pair_summary(rows, entry['pairs'], cfg.thresholds, cfg.norm_ratio)
        base['unrelated_length_boundary_matched'] = pair_summary(rows, entry['controls'], cfg.thresholds, cfg.norm_ratio)
        report['strata'].append(base)
    report.update(status='passed', memory_after=memory())
    write_json(cfg.worker_report, report)


def plan(cfg):
    _, keys, shape, source = source_info(cfg)
    rounded = source if cfg.encoding == 'fp32' else cfg.output_dir/(cfg.encoding+'-rounded.f32')
    if not rounded.exists():
        with rounded.open('xb') as stream:
            for chunk in read_chunks(source, shape[1], cfg.chunk_rows):
                stream.write(decode_rows(encode_rows(chunk, cfg.encoding), shape[1], cfg.encoding).tobytes())
    started = time.perf_counter()
    with StructuralPlan(rounded.read_bytes(), shape[1], dict(enumerate(keys)), cfg.structural_library,
                        mode='layout' if cfg.policy=='layout' else 'joint',
                        cosine=cfg.threshold, norm_ratio=cfg.norm_ratio) as planned:
        mapping = np.asarray(planned.representatives, dtype='<u8')
        order = np.asarray(planned.row_order, dtype='<u8')
        mapping.tofile(cfg.map_file)
        order.tofile(cfg.order_file)
        if cfg.policy == 'layout':
            planned.save_index(cfg.output_dir/'keys.das')
            planned.save_index(cfg.output_dir/'keys.flat', index='flat')
            # Native build verifies all key IDs; also reopen its serving index.
            with StructuralIndex(cfg.output_dir/'keys.das', cfg.structural_library) as index:
                for start in range(0, len(keys), cfg.chunk_rows):
                    if index.lookup(keys[start:start+cfg.chunk_rows]) != list(range(start,min(start+cfg.chunk_rows,len(keys)))):
                        raise AssertionError('fresh DATrie original-ID lookup mismatch')
        write_json(cfg.worker_report, dict(status='passed', **software_environment(), encoding=cfg.encoding,
            policy=cfg.policy, threshold=cfg.threshold, norm_ratio=cfg.norm_ratio,
            merged_rows=int(np.sum(mapping!=np.arange(shape[0]))),
            mapping_sha256=file_sha256(cfg.map_file), order_sha256=file_sha256(cfg.order_file),
            rounded_sha256=file_sha256(rounded), original_bytes_preserved=planned.original_bytes_preserved,
            plan_seconds=time.perf_counter()-started, memory_after=memory()))


def maps(cfg):
    return None if not cfg.map_file else np.fromfile(cfg.map_file, dtype='<u8').astype(np.int64)


def build(cfg):
    _, _, shape, source = source_info(cfg)
    mapping = maps(cfg)
    order = None if not cfg.order_file else np.fromfile(cfg.order_file,dtype='<u8').tolist()
    started = time.perf_counter()
    metadata = pack_quantized_file(source, cfg.bundle, shape[1], cfg.encoding,
        native=cfg.policy!='raw', representatives=None if mapping is None else mapping.tolist(),
        row_order=order, deduplicate=cfg.policy=='encoded-dedup',
        max_block_bytes=cfg.max_block_bytes, chunk_rows=cfg.chunk_rows)
    structural = dict(policy=cfg.policy, minimum_cosine=cfg.threshold if cfg.policy=='joint' else None,
        minimum_norm_ratio=cfg.norm_ratio if cfg.policy=='joint' else None,
        planner='production DATrie + mark-compact' if mapping is not None else None,
        mapping_sha256=None if cfg.map_file is None else file_sha256(cfg.map_file),
        contract='original key IDs retained; joint changes vectors; map is inside table.men')
    write_json(cfg.bundle/'structure.json', structural)
    write_json(cfg.worker_report, dict(status='passed', **software_environment(), metadata=metadata,
        structure=structural, build_seconds=time.perf_counter()-started,
        bundle_bytes=sum(p.stat().st_size for p in cfg.bundle.iterdir()), memory_after=memory()))


def verify(cfg):
    _, keys, shape, source = source_info(cfg)
    rows = np.memmap(source, mode='r', dtype='<f4', shape=shape)
    mapping = maps(cfg)
    if mapping is None:
        mapping = np.arange(shape[0])
    if len(mapping)!=shape[0] or np.any(mapping<0) or np.any(mapping>=shape[0]) or not np.array_equal(mapping[mapping],mapping):
        raise ValueError('invalid direct representative map')
    orders = np.asarray([len(k) for k in keys])
    squared_error = np.zeros(shape[0]); squared_source = np.zeros(shape[0])
    added_error = np.zeros(shape[0]); cosines = np.zeros(shape[0]); ratios = np.zeros(shape[0])
    rel = np.zeros(shape[0]); rel_added = np.zeros(shape[0]); rounded_norm = np.zeros(shape[0])
    encoded_hash, decoded_hash = hashlib.sha256(), hashlib.sha256()
    started = time.perf_counter()
    with QuantizedRows(cfg.bundle, cfg.native_library, cache_bytes=0) as reader:
        for start in range(0,shape[0],cfg.chunk_rows):
            stop=min(start+cfg.chunk_rows,shape[0]); ids=np.arange(start,stop,dtype='<u8')
            original = rows[start:stop]
            oracle = encode_rows(rows[mapping[start:stop]], cfg.encoding)
            actual = reader.encoded_rows(ids)
            if actual.tobytes()!=oracle.tobytes():
                raise AssertionError('original-ID encoded representative bytes differ')
            output=np.empty((stop-start,shape[1]),dtype='<f4')
            reader.gather_into(ids, output)
            if output.tobytes()!=decode_rows(oracle,shape[1],cfg.encoding).tobytes():
                raise AssertionError('decoded output bytes differ')
            encoded_hash.update(actual.tobytes()); decoded_hash.update(output.tobytes())
            m=row_metrics(original,output)
            rounded=decode_rows(encode_rows(original,cfg.encoding),shape[1],cfg.encoding)
            extra=row_metrics(rounded,output)
            squared_error[start:stop]=m['square_error']; squared_source[start:stop]=m['square_source']
            cosines[start:stop]=m['cosine']; ratios[start:stop]=m['norm_ratio']; rel[start:stop]=m['relative_l2']
            added_error[start:stop]=extra['square_error']; rounded_norm[start:stop]=extra['square_source']
            rel_added[start:stop]=extra['relative_l2']
            changed=mapping[start:stop]!=ids
            if cfg.policy=='joint':
                if np.any(extra['cosine'][changed] < cfg.threshold-1e-7) or np.any(extra['norm_ratio'][changed]<cfg.norm_ratio-1e-7):
                    raise AssertionError('final representative fails direct joint guards')
                for i,r in zip(ids[changed],mapping[start:stop][changed]):
                    if not related(keys[i],keys[r]):
                        raise AssertionError('representative lacks real prefix/suffix relation')
        if decoded_hash.hexdigest()!=reader.metadata['decoded_float32_sha256']:
            raise AssertionError('full decoded hash disagrees with export')
        bag_metrics=None
        if (cfg.output_dir/'bags.json').exists():
            bags=json.loads((cfg.output_dir/'bags.json').read_text())
            original_words=[]; quantized_words=[]; actual_words=[]
            for ids in bags:
                original_words.append(rows[ids].astype(np.float64).mean(0))
                quantized_words.append(decode_rows(encode_rows(rows[ids],cfg.encoding),shape[1],cfg.encoding).astype(np.float64).mean(0))
                # Every physical/forwarded row was verified above. Composition
                # uses that same oracle, without mixing query timing into it.
                actual_words.append(decode_rows(encode_rows(rows[mapping[ids]],cfg.encoding),shape[1],cfg.encoding).astype(np.float64).mean(0))
            bag_metrics=dict(vs_original=fidelity(original_words,actual_words),
                             vs_quantization_only=fidelity(quantized_words,actual_words))
    changed=mapping!=np.arange(shape[0])
    def metrics(mask):
        return dict(rows=int(mask.sum()), merged_rows=int((changed & mask).sum()),
            relative_frobenius=float(np.sqrt(squared_error[mask].sum()/max(squared_source[mask].sum(),np.finfo(np.float64).tiny))),
            added_relative_frobenius_vs_quantization=float(np.sqrt(added_error[mask].sum()/max(rounded_norm[mask].sum(),np.finfo(np.float64).tiny))),
            cosine=distribution(cosines[mask]), norm_ratio=distribution(ratios[mask]),
            relative_l2=distribution(rel[mask]), added_relative_l2=distribution(rel_added[mask]))
    trace=np.fromfile(cfg.training_dir/'queries.u64',dtype='<u8')
    mass=np.bincount(trace.astype(np.int64),minlength=shape[0])
    result=dict(status='passed', **software_environment(), all_rows_verified=shape[0],
        all_encoded_and_decoded_original_ids_byte_exact_to_representative=True,
        encoded_original_id_sha256=encoded_hash.hexdigest(), decoded_original_id_sha256=decoded_hash.hexdigest(),
        overall=metrics(np.ones(shape[0],dtype=bool)), merged_only=metrics(changed),
        by_order={str(int(n)):metrics(orders==n) for n in np.unique(orders)},
        heldout_query_weighted_relative_frobenius=float(np.sqrt(np.sum(mass*squared_error)/max(np.sum(mass*squared_source),np.finfo(np.float64).tiny))),
        heldout_query_weighted_added_frobenius=float(np.sqrt(np.sum(mass*added_error)/max(np.sum(mass*rounded_norm),np.finfo(np.float64).tiny))),
        composed_ngram_mean=bag_metrics, verify_seconds=time.perf_counter()-started, memory_after=memory())
    write_json(cfg.worker_report,result)


def query(cfg):
    # Numeric serving does not need the lexical registry or offline full map.
    # Construct the oracle first, then release its map/source before RSS/open.
    training=json.loads((cfg.training_dir/'training.json').read_text())
    shape=(training['ngram_rows'],training['dim'])
    source=cfg.training_dir/'trained.f32'
    rows=np.memmap(source,mode='r',dtype='<f4',shape=shape)
    ids=np.fromfile(cfg.output_dir/(cfg.case+'.u64'),dtype='<u8')
    mapping=maps(cfg)
    selected=ids if mapping is None else mapping[ids]
    expected=decode_rows(encode_rows(rows[selected],cfg.encoding),shape[1],cfg.encoding)
    del mapping, selected
    batches=[np.ascontiguousarray(ids[i:i+cfg.batch_rows]) for i in range(0,len(ids),cfg.batch_rows)]
    # Release oracle source pages before opening the measured backend.
    rows._mmap.close()
    output=np.empty((cfg.batch_rows,shape[1]),dtype='<f4')
    scratch=np.empty((cfg.batch_rows,row_width(shape[1],cfg.encoding)),dtype='u1')
    result=dict(status='running', **software_environment(), memory_before=memory(),
        output_and_scratch_bytes=output.nbytes+scratch.nbytes,
        oracle_bytes=expected.nbytes, trace_rows=len(ids), batch_rows=cfg.batch_rows,
        trace_sha256=file_sha256(cfg.output_dir/(cfg.case+'.u64')))
    with QuantizedRows(cfg.bundle,cfg.native_library,cache_bytes=cfg.cache_bytes) as reader:
        result['memory_open']=memory()
        def replay(chosen, offset_check=True):
            wall=[]; cpu=[]
            for i,batch in enumerate(chosen):
                t=time.perf_counter_ns(); c=time.process_time_ns()
                reader.gather_into(batch,output[:len(batch)],encoded_scratch=scratch[:len(batch)])
                wall.append((time.perf_counter_ns()-t)/1e6); cpu.append((time.process_time_ns()-c)/1e6)
                if offset_check and output[:len(batch)].tobytes()!=expected[i*cfg.batch_rows:i*cfg.batch_rows+len(batch)].tobytes():
                    raise AssertionError('timed gather oracle mismatch')
            return dict(wall=samples(wall),cpu=samples(cpu))
        result['first_trace']=replay(batches)
        result['memory_first']=memory()
        result['second_trace']=replay(batches)
        result['memory_second']=memory()
        hot=batches[:2]
        replay(hot)
        result['hot_trace']=replay(hot*cfg.repeat,offset_check=False)
        result['memory_hot']=memory()
    result.update(status='passed',all_first_and_second_query_bytes_exact=True)
    write_json(cfg.worker_report,result)


def run(cfg):
    prepare(cfg)
    root=Path(__file__).resolve().parents[1]
    info=json.loads((cfg.output_dir/'provenance.json').read_text())
    report=dict(status='running',provenance=info,workers=[],variants=[])
    def worker(phase, name, **options):
        out=cfg.output_dir/(name+'.json'); log=cfg.output_dir/(name+'.log')
        command=[sys.executable,'-m','tools.benchmark_large_structure',phase,
                 '--training-dir',str(cfg.training_dir),'--output-dir',str(cfg.output_dir),
                 '--native-library',str(cfg.native_library),'--structural-library',str(cfg.structural_library),
                 '--worker-report',str(out), '--thresholds',','.join(map(str,cfg.thresholds)),
                 '--norm-ratio',str(cfg.norm_ratio),'--sample-per-order',str(cfg.sample_per_order),
                 '--seed',str(cfg.seed),'--max-block-bytes',str(cfg.max_block_bytes),
                 '--chunk-rows',str(cfg.chunk_rows),'--cache-bytes',str(cfg.cache_bytes),
                 '--batch-rows',str(cfg.batch_rows),'--repeat',str(cfg.repeat)]
        for k,v in options.items():
            if v is not None:
                command.extend(['--'+k.replace('_','-'),str(v)])
        started=time.perf_counter()
        with log.open('w') as stream:
            process=subprocess.run(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT)
        entry=dict(phase=phase,name=name,command=shlex.join(command),returncode=process.returncode,
            wall_seconds=time.perf_counter()-started,log_sha256=file_sha256(log),
            result=json.loads(out.read_text()) if out.exists() else None)
        report['workers'].append(entry)
        write_json(cfg.output_dir/'report.json',report)
        print(json.dumps(dict(phase=phase,name=name,returncode=process.returncode,seconds=entry['wall_seconds'])),flush=True)
        if process.returncode:
            raise RuntimeError('worker failed; inspect '+str(log))
        return entry['result']
    worker('audit','pair-audit')
    for encoding in cfg.encodings:
        layout=encoding+'-layout'
        worker('plan','plan-'+layout,encoding=encoding,policy='layout',
            map_file=cfg.output_dir/(layout+'.map'),order_file=cfg.output_dir/(layout+'.order'))
        policies=[('raw',None,None,None),('identity',None,None,None),
                  ('layout',None,None,cfg.output_dir/(layout+'.order')),('encoded-dedup',None,None,None)]
        for threshold in cfg.thresholds:
            label=encoding+'-joint-'+str(threshold)
            mapping=cfg.output_dir/(label+'.map'); order=cfg.output_dir/(label+'.order')
            worker('plan','plan-'+label,encoding=encoding,policy='joint',threshold=threshold,map_file=mapping,order_file=order)
            policies.append(('joint',threshold,mapping,order))
        for policy,threshold,mapping,order in policies:
            name=encoding+'-'+policy+('' if threshold is None else '-'+str(threshold))
            bundle=cfg.output_dir/name
            options=dict(encoding=encoding,policy=policy,threshold=threshold,map_file=mapping,order_file=order,bundle=bundle)
            built=worker('build','build-'+name,**options)
            if built['metadata']['source_float32_sha256'] != info['source_sha256']:
                raise AssertionError('export source differs from registered trained checkpoint')
            verified=worker('verify','verify-'+name,**options)
            variant=dict(name=name,encoding=encoding,policy=policy,threshold=threshold,
                build=built,verification=verified,options={k:str(v) if isinstance(v,Path) else v for k,v in options.items()})
            report['variants'].append(variant)
            write_json(cfg.output_dir/'report.json',report)
    rng=random.Random(cfg.seed)
    for round_id in range(cfg.rounds):
        jobs=[(v,case) for v in report['variants'] for case in ('heldout-lexical','uniform-id-stress')]
        rng.shuffle(jobs)
        for variant,case in jobs:
            worker('query',f"query-{variant['name']}-{case}-{round_id}",case=case,**variant['options'])
    report.update(status='passed',index_bytes={name:(cfg.output_dir/name).stat().st_size for name in ('keys.das','keys.flat')})
    write_json(cfg.output_dir/'report.json',report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('run','audit','plan','build','verify','query'))
    for name in ('training-dir','output-dir','native-library','structural-library'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('worker-report','map-file','order-file','bundle','corpus-zip'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--encodings',default='fp32,fp16,int8')
    p.add_argument('--encoding',choices=('fp32','fp16','int8'),default='fp32')
    p.add_argument('--policy',choices=('raw','identity','layout','encoded-dedup','joint'),default='raw')
    p.add_argument('--thresholds',default='.99,.98,.97,.95,.90')
    p.add_argument('--threshold',type=float,default=.99)
    p.add_argument('--norm-ratio',type=float,default=.99)
    p.add_argument('--sample-per-order',type=int,default=4096)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--trace-rows',type=int,default=8192)
    p.add_argument('--bag-count',type=int,default=1024)
    p.add_argument('--batch-rows',type=int,default=512)
    p.add_argument('--rounds',type=int,default=2)
    p.add_argument('--repeat',type=int,default=10)
    p.add_argument('--chunk-rows',type=int,default=4096)
    p.add_argument('--cache-bytes',type=int,default=4194304)
    p.add_argument('--max-block-bytes',type=int,default=8192)
    p.add_argument('--case',choices=('heldout-lexical','uniform-id-stress'),default='heldout-lexical')
    cfg=p.parse_args()
    cfg.thresholds=[float(x) for x in cfg.thresholds.split(',')]
    cfg.encodings=cfg.encodings.split(',')
    if (not cfg.thresholds or any(not 0<x<=1 for x in cfg.thresholds) or not 0<cfg.norm_ratio<=1
        or not 0<cfg.threshold<=1 or any(x not in ('fp32','fp16','int8') for x in cfg.encodings)
        or min(cfg.trace_rows,cfg.sample_per_order,cfg.bag_count,cfg.batch_rows,cfg.rounds,cfg.repeat,cfg.chunk_rows,cfg.max_block_bytes)<1
        or cfg.cache_bytes<0 or cfg.trace_rows%cfg.batch_rows):
        p.error('invalid thresholds, encodings, buffer sizes or trace length')
    if cfg.phase!='run' and cfg.worker_report is None:
        p.error('worker report required')
    if cfg.phase in ('build','verify','query') and cfg.bundle is None:
        p.error('bundle required')
    if cfg.phase=='plan' and (cfg.map_file is None or cfg.order_file is None):
        p.error('map and order outputs required')
    if cfg.phase=='run' and os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        p.error('explicit CUDA_VISIBLE_DEVICES="" required for CPU run')
    globals()[cfg.phase](cfg)


if __name__=='__main__':
    main()
