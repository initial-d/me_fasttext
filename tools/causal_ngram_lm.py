"""Small causal character LM with an exact learned n-gram memory.

Reference architecture: character embedding + mean of known suffix memories,
then a one-layer GRU and full-vocabulary softmax. Not an Engram implementation.
Lookup keys/vocabulary fit training text only; missing keys map to zero memory.
All suffixes end at the input character, before the predicted next character.
"""
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def fit_registry(text, max_order, min_count=3):
    if type(max_order) is not int or max_order < 2 or min_count < 1 or not text:
        raise ValueError("nonempty training text, order >= 2, positive count required")
    characters = sorted(set(text))
    counts = Counter(text[i - length + 1:i + 1] for i in range(len(text))
                     for length in range(2, min(max_order, i + 1) + 1))
    grams = sorted(g for g, count in counts.items() if count >= min_count)
    if not grams:
        raise ValueError("no repeated training n-grams; lower min_count or use more text")
    return {"characters": characters, "grams": grams, "max_order": max_order,
            "min_count": min_count}


def encode_segment(text, registry):
    """No future-character lookup and no context crossing a segment boundary."""
    vocabulary = {char: i + 1 for i, char in enumerate(registry["characters"])}
    lookup = {gram: i + 1 for i, gram in enumerate(registry["grams"])}
    chars = np.asarray([vocabulary.get(char, 0) for char in text], dtype=np.int64)
    ids = np.zeros((len(text), registry["max_order"] - 1), dtype=np.int64)
    for i in range(len(text)):
        for length in range(2, min(registry["max_order"], i + 1) + 1):
            ids[i, length - 2] = lookup.get(text[i - length + 1:i + 1], 0)
    return {"characters": torch.from_numpy(chars), "memory_ids": torch.from_numpy(ids),
            "unknown_characters": int((chars == 0).sum()),
            "known_memory_slot_fraction": float((ids != 0).mean()),
            "positions_with_memory_fraction": float((ids != 0).any(axis=1).mean())}


class CausalNgramLM(nn.Module):
    def __init__(self, vocabulary_size, memory_rows, dim=64):
        super().__init__()
        if min(vocabulary_size, memory_rows, dim) < 1:
            raise ValueError("positive model dimensions required")
        self.dim = dim
        self.characters = nn.Embedding(vocabulary_size, dim)
        self.memory = nn.Embedding(memory_rows + 1, dim, sparse=True, padding_idx=0)
        self.recurrent = nn.GRU(dim, dim, batch_first=True)
        self.output = nn.Linear(dim, vocabulary_size)
        with torch.no_grad():
            self.characters.weight.uniform_(-1 / dim, 1 / dim)
            self.memory.weight.uniform_(-1 / dim, 1 / dim)
            self.memory.weight[0].zero_()

    def forward(self, characters, memory_ids, hidden=None, use_memory=True):
        values = self.characters(characters)
        if use_memory:
            count = memory_ids.ne(0).sum(-1, keepdim=True).clamp_min(1)
            values = values + self.memory(memory_ids).sum(-2) / count
        states, hidden = self.recurrent(values, hidden)
        return self.output(states), hidden


def train_reference(model, segment, steps=1200, batch_size=16, sequence_length=32,
                    dense_lr=.002, memory_lr=.01, seed=3):
    if min(steps, batch_size, sequence_length) < 1 or len(segment["characters"]) <= sequence_length:
        raise ValueError("positive training dimensions and enough training characters required")
    if dense_lr <= 0 or memory_lr <= 0:
        raise ValueError("positive learning rates required")
    rng = np.random.default_rng(seed)
    dense = [parameter for name, parameter in model.named_parameters() if name != "memory.weight"]
    dense_optimizer = torch.optim.Adam(dense, lr=dense_lr)
    memory_optimizer = torch.optim.SparseAdam([model.memory.weight], lr=memory_lr)
    offsets = torch.arange(sequence_length)[None, :]
    history = []
    model.train()
    for step in range(steps):
        starts = torch.from_numpy(rng.integers(len(segment["characters"]) - sequence_length,
                                             size=batch_size))
        positions = starts[:, None] + offsets
        inputs = segment["characters"][positions]
        targets = segment["characters"][positions + 1]
        logits, _ = model(inputs, segment["memory_ids"][positions])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        dense_optimizer.zero_grad(set_to_none=True)
        memory_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dense, 1.0)
        dense_optimizer.step()
        memory_optimizer.step()
        if step % 100 == 0 or step == steps - 1:
            history.append({"step": step, "cross_entropy_nats": float(loss.detach())})
    model.eval()
    return history


def evaluate(model, segment, burn_in=64, chunk_size=256, use_memory=True):
    """Stream the whole segment with recurrent state, reset only at its start.

    Score all next-character targets after a fixed burn-in, including unknown
    characters (ID 0). Chunk boundaries do not reset the GRU or n-gram context.
    """
    length = len(segment["characters"]) - 1
    if burn_in < 0 or chunk_size < 1 or length <= burn_in:
        raise ValueError("valid chunk/burn-in and a nonempty evaluation region required")
    predictions, hidden = [], None
    with torch.no_grad():
        for start in range(0, length, chunk_size):
            end = min(start + chunk_size, length)
            logits, hidden = model(segment["characters"][None, start:end],
                                   segment["memory_ids"][None, start:end], hidden, use_memory)
            predictions.append(logits.squeeze(0))
    logits = torch.cat(predictions)[burn_in:]
    targets = segment["characters"][burn_in + 1:]
    loss = F.cross_entropy(logits, targets, reduction="none")
    top1 = logits.argmax(-1)
    return {"logits": logits, "targets": targets, "loss": loss.numpy().astype(np.float64),
            "correct": top1.eq(targets).numpy(), "top1": top1.numpy(),
            "positions": np.arange(burn_in, length, dtype=np.int64)}


def metric_summary(scores):
    loss = float(np.mean(scores["loss"]))
    return {"targets": len(scores["loss"]), "cross_entropy_nats": loss,
            "character_perplexity": float(np.exp(loss)), "next_character_accuracy": float(np.mean(scores["correct"])),
            "unknown_targets": int(scores["targets"].eq(0).sum())}


def prediction_delta(candidate, reference):
    if not torch.equal(candidate["targets"], reference["targets"]):
        raise ValueError("comparison targets differ")
    # Double precision KL limits cancellation for almost identical float32 logits.
    logp = F.log_softmax(reference["logits"].double(), -1)
    logq = F.log_softmax(candidate["logits"].double(), -1)
    kl = (logp.exp() * (logp - logq)).sum(-1).clamp_min(0)
    return {"perplexity_relative_change": float(np.expm1(np.mean(candidate["loss"] - reference["loss"]))),
            "mean_prediction_kl_nats": float(kl.mean()),
            "prediction_kl_p99_nats": float(torch.quantile(kl, .99)),
            "maximum_prediction_kl_nats": float(kl.max()),
            "top1_changed_fraction": float(np.mean(candidate["top1"] != reference["top1"])),
            "accuracy_change": float(np.mean(candidate["correct"]) - np.mean(reference["correct"])),
            "maximum_absolute_logit_change": float((candidate["logits"] - reference["logits"]).abs().max()),
            "maximum_per_target_cross_entropy_increase": float(np.max(candidate["loss"] - reference["loss"]))}
