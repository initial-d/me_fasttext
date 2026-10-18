"""Train exact character-n-gram SGNS and audit structural/co-update similarity.

Public-corpus reference experiment, not the legacy C++ runner or Engram training.
Input rows begin independent uniform(-1/d,1/d), following FastText-style scale.
Rows in one word bag receive the same SGD gradient; shared-update signatures are
measured explicitly rather than assuming all prefix/suffix neighbors co-train.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

try:
    from .native_row_store import pack_native, NativeRows
except ImportError:
    from native_row_store import pack_native, NativeRows


def ngrams(word, maximum):
    bounded = "<" + word + ">"
    return [bounded[i:i + n] for i in range(len(bounded))
            for n in range(2, maximum + 1) if i + n <= len(bounded)]


def stats(values, exact=0):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"pairs": 0}
    return {"pairs": len(values), "mean_cosine": float(values.mean()),
            "min_cosine": float(values.min()), "p10_cosine": float(np.quantile(values, 0.1)),
            "median_cosine": float(np.median(values)), "p90_cosine": float(np.quantile(values, 0.9)),
            "fraction_ge_0_99": float((values >= 0.99).mean()),
            "fraction_ge_0_999": float((values >= 0.999).mean()), "byte_identical_pairs": exact}


def prepare(text, maximum, vocab_limit, min_count):
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
    counts = Counter(tokens)
    words = sorted((w for w, c in counts.items() if c >= min_count), key=lambda w: (-counts[w], w))[:vocab_limit]
    ids = {word: i for i, word in enumerate(words)}
    seq = [ids.get(w, -1) for w in tokens]
    bags = [ngrams(w, maximum) for w in words]
    grams = sorted({g for bag in bags for g in bag})
    gram_ids = {g: i for i, g in enumerate(grams)}
    signatures = defaultdict(list)
    for word_id, bag in enumerate(bags):
        for gram, multiplicity in Counter(bag).items():
            signatures[gram_ids[gram]].append((word_id, multiplicity))
    # Adjacent original token positions, without collapsing OOV gaps.
    pairs = [(seq[i], seq[j]) for i in range(len(seq)) if seq[i] >= 0
             for j in range(max(0, i - 2), min(len(seq), i + 3))
             if i != j and seq[j] >= 0]
    max_bag = max(len(bag) + 1 for bag in bags)
    padded = torch.zeros((len(words), max_bag), dtype=torch.long)
    lengths = torch.empty(len(words), dtype=torch.long)
    # ID zero is padding, then word rows, then exact ngram rows.
    for i, bag in enumerate(bags):
        row = [i + 1] + [1 + len(words) + gram_ids[g] for g in bag]
        padded[i, :len(row)] = torch.tensor(row)
        lengths[i] = len(row)
    return words, grams, signatures, np.asarray(pairs, dtype=np.int64), padded, lengths, counts


def run(text, cfg, maximum, seed, output_dir):
    words, grams, signatures, pairs, bags, lengths, counts = prepare(text, maximum, cfg.vocab_limit, cfg.min_count)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    inputs = nn.EmbeddingBag(1 + len(words) + len(grams), cfg.dim, mode="sum", sparse=True, padding_idx=0)
    outputs = nn.Embedding(len(words), cfg.dim, sparse=True)
    with torch.no_grad():
        inputs.weight.uniform_(-1 / cfg.dim, 1 / cfg.dim)
        inputs.weight[0].zero_()
        outputs.weight.zero_()
    initial = inputs.weight.detach()[1 + len(words):].clone()
    optimizer = torch.optim.SGD(list(inputs.parameters()) + list(outputs.parameters()), lr=cfg.lr)
    negative_probs = torch.tensor([counts[w] ** 0.5 for w in words], dtype=torch.float32)
    negative_probs /= negative_probs.sum()
    losses, start = [], time.perf_counter()
    for step in range(cfg.steps):
        sample = pairs[rng.integers(0, len(pairs), size=cfg.batch_size)]
        center, target = torch.from_numpy(sample[:, 0].copy()), torch.from_numpy(sample[:, 1].copy())
        summed = inputs(bags[center])
        averaged = summed / lengths[center].unsqueeze(1)
        # The legacy FastText Model::update adds grad_ to every input row
        # without dividing by bag size, even though its hidden state is averaged.
        # Preserve that forward/backward convention rather than silently using
        # the standard autograd derivative of an averaged embedding bag.
        hidden = summed + (averaged - summed).detach()
        negative = torch.multinomial(negative_probs, cfg.batch_size * cfg.negatives, replacement=True).reshape(cfg.batch_size, cfg.negatives)
        positive_logit = (hidden * outputs(target)).sum(1)
        negative_logits = (hidden.unsqueeze(1) * outputs(negative)).sum(2)
        loss = (-torch.nn.functional.logsigmoid(positive_logit)
                - torch.nn.functional.logsigmoid(-negative_logits).sum(1)).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == cfg.steps - 1:
            losses.append({"step": step, "loss_per_pair": float(loss.detach()) / cfg.batch_size})
    learned = inputs.weight.detach()[1 + len(words):].contiguous()
    normalized = torch.nn.functional.normalize(learned, dim=1)
    lookup = {g: i for i, g in enumerate(grams)}
    prefix, suffix = [], []
    for child_id, gram in enumerate(grams):
        if gram[:-1] in lookup:
            prefix.append((lookup[gram[:-1]], child_id))
        if gram[1:] in lookup:
            suffix.append((lookup[gram[1:]], child_id))
    structural = sorted(set(prefix + suffix))
    same_signature = [p for p in structural if signatures[p[0]] == signatures[p[1]]]
    shared_words = [p for p in structural if set(w for w, _ in signatures[p[0]]) & set(w for w, _ in signatures[p[1]])]
    def summarize(pair_list):
        if not pair_list:
            return {"pairs": 0}
        index = torch.tensor(pair_list)
        cosine = (normalized[index[:, 0]] * normalized[index[:, 1]]).sum(1).numpy()
        raw = learned.view(torch.uint8).numpy()
        return stats(cosine, sum(np.array_equal(raw[a], raw[b]) for a, b in pair_list))
    result = {"max_order": maximum, "seed": seed, "vocabulary": len(words), "ngram_rows": len(grams),
              "training_pair_pool": len(pairs), "sampled_training_pairs": cfg.steps * cfg.batch_size,
              "training_seconds": time.perf_counter() - start, "loss_samples": losses,
              "prefix_parent": summarize(prefix), "suffix_parent": summarize(suffix),
              "all_structural": summarize(structural), "shared_host_word": summarize(shared_words),
              "identical_host_multiplicity_signature": summarize(same_signature)}
    if structural:
        pair_ids = torch.tensor(structural)
        init_normalized = torch.nn.functional.normalize(initial, dim=1)
        result["initial_structural"] = stats((init_normalized[pair_ids[:, 0]] * init_normalized[pair_ids[:, 1]]).sum(1).numpy())
    host_mass = {i: sum(counts[words[w]] * multiplicity for w, multiplicity in signature)
                 for i, signature in signatures.items()}
    result["structural_by_min_host_occurrences"] = {
        str(threshold): summarize([p for p in structural if min(host_mass[p[0]], host_mass[p[1]]) >= threshold])
        for threshold in (5, 50, 200, 1000)
    }
    # Positive principle check: identical update schedules beyond mere spelling.
    groups = defaultdict(list)
    for row_id, signature in signatures.items():
        groups[tuple(signature)].append(row_id)
    group_pairs = [(group[0], i) for group in groups.values() for i in group[1:]]
    result["all_identical_update_signatures"] = summarize(group_pairs)
    structural_index = torch.tensor(structural)
    if len(structural):
        norms = learned.norm(dim=1)
        norm_ratio = torch.minimum(norms[structural_index[:, 0]], norms[structural_index[:, 1]]) / torch.maximum(norms[structural_index[:, 0]], norms[structural_index[:, 1]]).clamp_min(1e-30)
        cos = (normalized[structural_index[:, 0]] * normalized[structural_index[:, 1]]).sum(1)
        accepts = norm_ratio >= 0.99
        result["norm_ratio_audit"] = {"accepted": int(accepts.sum()),
                                      "accepted_but_cosine_below_0_99": int((accepts & (cos < 0.99)).sum())}
    directory = output_dir / f"order-{maximum}-seed-{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = directory / "trained-ngram.f32"
    raw = learned.view(torch.uint8).numpy().tobytes()
    raw_path.write_bytes(raw)
    result["learned_row_sha256"] = hashlib.sha256(raw).hexdigest()
    result["training_reference"] = "exact char-ngram bag + word row, SGNS 5 negatives, sparse SGD; not original C++ trainer"
    # No approximate merge: trained vectors stay byte exact in native storage.
    artifact = directory / "trained-ngram.men"
    packed = pack_native(raw_path, artifact, cfg.dim * 4, 64, deduplicate=True)
    with NativeRows(artifact, cfg.native_library) as native:
        assert native.rows(range(len(grams))) == raw
    result["native_lossless_storage"] = packed
    result["native_lossless_storage"]["all_rows_byte_exact"] = True
    result["native_lossless_storage"]["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    # Shuffling row order preserves bytes/shape but destroys lexical locality.
    shuffled_raw = learned[torch.from_numpy(rng.permutation(len(grams)))].contiguous().view(torch.uint8).numpy().tobytes()
    shuffled_path = directory / "trained-shuffled.f32"
    shuffled_path.write_bytes(shuffled_raw)
    shuffled_artifact = directory / "trained-shuffled.men"
    shuffled = pack_native(shuffled_path, shuffled_artifact, cfg.dim * 4, 64, deduplicate=True)
    with NativeRows(shuffled_artifact, cfg.native_library) as native:
        assert native.rows(range(len(grams))) == shuffled_raw
    result["shuffled_lossless_storage"] = shuffled
    result["shuffled_lossless_storage"]["all_rows_byte_exact"] = True
    result["training_settings_note"] = "forward average / unscaled input-row gradient follows legacy SGNS; negative counts exponent 0.5; batched updates, no C++ subsampling/learning-rate schedule"
    examples = []
    if len(same_signature):
        for a, b in same_signature[:10]:
            examples.append({"a": grams[a], "b": grams[b], "cosine": float((normalized[a] * normalized[b]).sum()),
                             "host_words": [words[w] for w, _ in signatures[a]][:12]})
    result["same_update_examples"] = examples
    print(json.dumps({k: result[k] for k in ("max_order", "seed", "ngram_rows", "all_structural", "identical_host_multiplicity_signature")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-orders", default="4,8,12")
    parser.add_argument("--seeds", default="3,7,11")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--vocab-limit", type=int, default=1200)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--negatives", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()
    if min(args.dim, args.vocab_limit, args.min_count, args.steps, args.batch_size, args.negatives) <= 0:
        parser.error("training dimensions must be positive")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = args.corpus.read_bytes()
    report = {"category": "public-text trained reference n-gram experiment; not Engram or original C++ reproduction",
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "command": " ".join([sys.executable, *sys.argv]), "corpus_sha256": hashlib.sha256(raw).hexdigest(),
              "corpus_bytes": len(raw), "python": sys.version, "pytorch": torch.__version__, "numpy": np.__version__,
              "platform": platform.platform(), "device": "cpu", "threads": 1,
              "settings": {k: v for k, v in vars(args).items() if not isinstance(v, Path)},
              "limitations": ["One small English public corpus; not the Chinese paper corpus.",
                              "Reference SGNS, fixed optimizer/settings and finite training; no task-quality or convergence claim.",
                              "Spelling-based structural pairs and identical update signatures are separate groups.",
                              "Cosine similarity does not prove byte equality or strict lossless row sharing."], "runs": []}
    for order in (int(x) for x in args.max_orders.split(",")):
        for seed in (int(x) for x in args.seeds.split(",")):
            report["runs"].append(run(raw.decode("utf-8"), args, order, seed, args.output_dir))
            (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
