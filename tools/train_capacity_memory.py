"""Train a larger exact n-gram SGNS storage reference on local public text.

No downloads or CUDA. Uses sparse CPU SGD and the same averaged-forward,
unaveraged-bag-gradient convention as train_ngram_similarity. This finite
storage reference is not a converged language model or trained Engram.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch
from torch import nn

from .train_ngram_similarity import ngrams
from .engram_row_adapter import file_sha256


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def prepare(raw, order, min_count):
    text = re.sub(r'<[^>]*>', ' ', raw.decode('utf-8', errors='replace'))
    tokens = [w for w in re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower()) if 2 <= len(w) <= 64]
    counts = Counter(tokens)
    words = sorted(w for w, count in counts.items() if count >= min_count)
    lookup = {w: i for i, w in enumerate(words)}
    sequence = np.fromiter((lookup.get(w, -1) for w in tokens), dtype=np.int64)
    bags = [ngrams(w, order) for w in words]
    grams = sorted({g for bag in bags for g in bag})
    gram_ids = {g: i for i, g in enumerate(grams)}
    offsets = np.zeros(len(words) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([1 + len(bag) for bag in bags])
    indices = np.empty(offsets[-1], dtype=np.int64)
    for i, bag in enumerate(bags):
        indices[offsets[i]:offsets[i+1]] = [i] + [len(words) + gram_ids[g] for g in bag]
    # Genuine adjacent token positions; OOV gaps are not collapsed. One
    # representative occurrence per center ensures vocabulary coverage.
    positions = np.flatnonzero((sequence[:-1] >= 0) & (sequence[1:] >= 0))
    _, first = np.unique(sequence[positions], return_index=True)
    coverage_positions = positions[first]
    return words, grams, sequence, positions, coverage_positions, indices, offsets, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus-zip', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--order', type=int, default=12)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--min-count', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--random-pairs', type=int, default=262144)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--negatives', type=int, default=5)
    parser.add_argument('--lr', type=float, default=.02)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exclude-untrained', action='store_true', help='explicitly export only updated rows, with source-ID provenance')
    args = parser.parse_args()
    if min(args.order-1, args.dim, args.min_count, args.epochs, args.batch_size, args.threads) < 1:
        parser.error('positive settings and order >= 2 required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('set CUDA_VISIBLE_DEVICES to the empty string for this CPU-only study')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    report_path = args.output_dir / 'training.json'
    start = time.perf_counter()
    report = {'status': 'preparing', 'category': 'public-text finite CPU SGNS storage reference',
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'command': shlex.join([sys.executable, *sys.argv]), 'settings': vars(args).copy(),
              'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
              'os': platform.platform(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'corpus_source': 'https://mattmahoney.net/dc/enwik8.zip',
              'corpus_license_note': 'Wikipedia-derived benchmark XML; publish aggregates/hashes only, no text or weights.',
              'zip_sha256': file_sha256(args.corpus_zip), 'loss_samples': [],
              'limitations': ['Finite reference training, not convergence or task quality.',
                              'ASCII word tokenization of XML-tag-stripped Wikipedia benchmark text.',
                              'No trained DeepSeek weights, no whole-model inference or GPU memory result.',
                              'Coverage sampling overweights rare words; it is recorded, not claimed as standard full-corpus SGNS.',
                              'Shared CPU host, monetary/energy cost unmeasured.']}
    report['settings'] = {k: str(v) if isinstance(v, Path) else v for k, v in report['settings'].items()}
    with zipfile.ZipFile(args.corpus_zip) as archive:
        raw = archive.read('enwik8')
    split = int(.9 * len(raw))
    report['corpus_raw_bytes'] = len(raw)
    report['corpus_raw_sha256'] = hashlib.sha256(raw).hexdigest()
    report['train_raw_sha256'] = hashlib.sha256(raw[:split]).hexdigest()
    report['heldout_raw_sha256'] = hashlib.sha256(raw[split:]).hexdigest()
    words, grams, seq, positions, coverage, indices, offsets, counts = prepare(raw[:split], args.order, args.min_count)
    del raw
    report.update({'vocabulary': len(words), 'ngram_rows': len(grams), 'dim': args.dim,
                   'raw_table_bytes': len(grams) * args.dim * 4, 'training_tokens': len(seq),
                   'valid_adjacent_pair_pool': len(positions), 'coverage_centers': len(coverage),
                   'status': 'training', 'preparation_seconds': time.perf_counter()-start})
    write_json(report_path, report)
    print(json.dumps({k: report[k] for k in ('vocabulary','ngram_rows','raw_table_bytes','coverage_centers')}), flush=True)
    inputs = nn.EmbeddingBag(len(words)+len(grams), args.dim, mode='sum', sparse=True, include_last_offset=True)
    outputs = nn.Embedding(len(words), args.dim, sparse=True)
    with torch.no_grad():
        inputs.weight.uniform_(-1/args.dim, 1/args.dim)
        outputs.weight.zero_()
    # Keep the initialization on disk, never fabricate larger duplicate rows.
    initial_path = args.output_dir / 'initial.f32'
    inputs.weight.detach()[len(words):].numpy().astype('<f4', copy=False).tofile(initial_path)
    optimizer = torch.optim.SGD([*inputs.parameters(), *outputs.parameters()], lr=args.lr)
    probs = torch.tensor([counts[w]**.5 for w in words], dtype=torch.float32)
    probs /= probs.sum()
    training_started = time.perf_counter()
    centers_seen = np.zeros(len(words), dtype=np.bool_)
    sampled_pairs = 0
    for epoch in range(args.epochs):
        sampled = np.concatenate([rng.permutation(coverage), rng.choice(positions, args.random_pairs)])
        rng.shuffle(sampled)
        losses = []
        for begin in range(0, len(sampled), args.batch_size):
            pos = sampled[begin:begin+args.batch_size]
            center = seq[pos]
            target = torch.from_numpy(seq[pos+1].copy())
            centers_seen[center] = True
            lengths = offsets[center+1]-offsets[center]
            batch_offsets = np.concatenate([[0], np.cumsum(lengths)])
            batch_ids = np.concatenate([indices[offsets[c]:offsets[c+1]] for c in center])
            summed = inputs(torch.from_numpy(batch_ids), torch.from_numpy(batch_offsets))
            averaged = summed / torch.from_numpy(lengths).unsqueeze(1)
            hidden = summed + (averaged-summed).detach()
            negative = torch.multinomial(probs, len(center)*args.negatives, replacement=True).reshape(len(center), args.negatives)
            positive = (hidden*outputs(target)).sum(1)
            negative_logits = (hidden.unsqueeze(1)*outputs(negative)).sum(2)
            loss = (-torch.nn.functional.logsigmoid(positive)-torch.nn.functional.logsigmoid(-negative_logits).sum(1)).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sampled_pairs += len(center)
            if begin % (args.batch_size*200) == 0:
                value = float(loss.detach()) / len(center)
                if not np.isfinite(value):
                    raise ValueError('nonfinite training loss')
                losses.append({'pair': sampled_pairs, 'loss_per_pair': value})
                print(json.dumps({'epoch':epoch, 'pairs':sampled_pairs, 'loss':value, 'seconds':time.perf_counter()-training_started}), flush=True)
        report['loss_samples'].extend(losses)
        report['completed_epochs'] = epoch+1
        report['sampled_training_pairs'] = sampled_pairs
        write_json(report_path, report)
    table = inputs.weight.detach()[len(words):].numpy()
    initial = np.memmap(initial_path, mode='r', dtype='<f4', shape=table.shape)
    changed = np.zeros(len(grams), dtype=np.bool_)
    finite = True
    for begin in range(0, len(grams), 4096):
        part = table[begin:begin+4096]
        changed[begin:begin+4096] = np.any(part.view('u4') != initial[begin:begin+4096].view('u4'), axis=1)
        finite &= bool(np.isfinite(part).all())
    if not finite or (not changed.all() and not args.exclude_untrained):
        report.update({'status':'incomplete-training-coverage','changed_rows':int(changed.sum()), 'all_finite':finite})
        write_json(report_path, report)
        raise ValueError('every published capacity row must change from initialization and be finite')
    report['registered_ngram_rows'] = len(grams)
    report['excluded_untrained_rows'] = int((~changed).sum())
    source_ids = np.flatnonzero(changed)
    source_ids.astype('<u8').tofile(args.output_dir / 'trained-source-ids.u64')
    # Publish a separately registered storage table of actual trained rows.
    # Keep its mapping to the original training registry explicit; this is not
    # a full-model converter and does not silently redirect model IDs.
    grams = [g for i,g in enumerate(grams) if changed[i]]
    path = args.output_dir / 'trained.f32'
    with path.open('wb') as dest:
        for begin in range(0,len(source_ids),4096):
            table[source_ids[begin:begin+4096]].astype('<f4',copy=False).tofile(dest)
    (args.output_dir / 'keys.json').write_text(json.dumps(grams) + '\n')
    # Held-out lexical bags refer only to registered original rows. No tiled IDs.
    with zipfile.ZipFile(args.corpus_zip) as archive:
        heldout = archive.read('enwik8')[split:].decode('utf-8', errors='replace')
    query_ids = []
    query_keys = []
    gram_ids = {g:i for i,g in enumerate(grams)}
    for word in re.findall(r"[a-z]+(?:'[a-z]+)?", re.sub(r'<[^>]*>', ' ', heldout).lower()):
        if not 2 <= len(word) <= 64:
            continue
        for gram in ngrams(word, args.order):
            if gram in gram_ids:
                query_ids.append(gram_ids[gram])
                query_keys.append(gram)
        if len(query_ids) >= 65536:
            break
    np.asarray(query_ids[:65536], dtype='<u8').tofile(args.output_dir / 'queries.u64')
    (args.output_dir / 'query-keys.json').write_text(json.dumps(query_keys[:65536])+'\n')
    report.update({'status':'passed', 'changed_rows':int(changed.sum()), 'all_rows_changed':True,
                   'ngram_rows':len(grams), 'raw_table_bytes':path.stat().st_size,
                   'source_row_ids_sha256':file_sha256(args.output_dir/'trained-source-ids.u64'),
                   'all_finite':True, 'center_coverage':int(centers_seen.sum()),
                   'training_seconds':time.perf_counter()-training_started,
                   'trained_sha256':file_sha256(path), 'keys_sha256':file_sha256(args.output_dir/'keys.json'),
                   'query_ids_sha256':file_sha256(args.output_dir/'queries.u64'),
                   'query_rows':min(65536,len(query_ids)), 'initial_sha256':file_sha256(initial_path)})
    write_json(report_path, report)
    print(json.dumps({'status':report['status'], 'rows':len(grams),'changed':int(changed.sum())}),flush=True)


if __name__ == '__main__':
    main()
