"""Frozen TinyEngram/Qwen3 checkpoints with native memory and explicit decode state.

The upstream Python files are supplied locally and verified against one pinned
revision. Loading/exporting never downloads a model or executes checkpoint code.
Supports unpadded eager inference and sequential greedy decode. CPU memory can
stage queried vectors to a fixed accelerator; CUDA requires hardware validation.
Backbone weights and tokenizer assets are included in the bundle.
"""
import argparse
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen3Config
from transformers.cache_utils import DynamicCache
import transformers

from .engram_row_adapter import (
    MappedMultiHeadEmbedding, export_multihead, file_sha256, head_offsets,
)
from .device_row_adapter import StagedMultiHeadEmbedding, _device

UPSTREAM_SHA = "317f23832aa8f8e382a0916ec469e81ba12b6dff"
SOURCE_HASHES = {
    "engram.py": "b5f20018ee7ff73d0b43911830b0e8cd4c4540038c7c7775b821a6969b666630",
    "engram_qwen.py": "9ff59ce9acaab626f6feb5aff44ccd3969257b4d11f62b9cedeb48c8030a59f6",
}
_decode_context = ContextVar("meft_tinyengram_decode", default=None)


def source_revision():
    """Report the source checkout, never an unrelated caller's working directory."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None  # Installed wheels retain exact source-file hashes instead.


def load_upstream(upstream_dir):
    """Isolate upstream globals so models can use different gate-step settings."""
    if transformers.__version__ != "4.57.3":
        raise ValueError("TinyEngram integration requires transformers==4.57.3")
    root = Path(upstream_dir)
    for name, digest in SOURCE_HASHES.items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"TinyEngram source differs from pinned revision: {name}")
    loaded = []
    previous = sys.modules.get("engram")
    try:
        for name in SOURCE_HASHES:
            module_name = "_meft_tiny_" + uuid.uuid4().hex
            spec = importlib.util.spec_from_file_location(module_name, root / name)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            finally:
                sys.modules.pop(module_name, None)
            loaded.append(module)
            if name == "engram.py":
                sys.modules["engram"] = module
    finally:
        if previous is None:
            sys.modules.pop("engram", None)
        else:
            sys.modules["engram"] = previous
    loaded[1].set_skip_engram(False)
    return tuple(loaded)


def _json(path):
    return json.loads(Path(path).read_text())


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _safe_path(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("bundle/checkpoint paths must stay within their directory")
    resolved = (Path(root) / path).resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise ValueError("bundle/checkpoint path escapes its directory")
    return resolved


def _checkpoint_keys(root):
    root = Path(root)
    index = root / "model.safetensors.index.json"
    if index.exists():
        declared = _json(index)["weight_map"]
        files = sorted(set(declared.values()))
    elif (root / "model.safetensors").exists():
        declared, files = None, ["model.safetensors"]
    else:
        raise ValueError("requires a full HF model.safetensors checkpoint (single or indexed shards)")
    keys = {}
    for name in files:
        with safe_open(_safe_path(root, name), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in keys:
                    raise ValueError(f"duplicate checkpoint tensor: {key}")
                keys[key] = name
    if declared is not None and keys != declared:
        raise ValueError("safetensors index differs from actual shard keys")
    return keys


def _tensor(root, keys, key):
    if key not in keys:
        raise ValueError(f"missing checkpoint tensor: {key}")
    with safe_open(_safe_path(root, keys[key]), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def _hash_lookup(mapping):
    return hashlib.sha256(mapping.compressed_tokenizer.lookup_table.tobytes()).hexdigest()


def _config(engram, values, tokenizer_dir, n_layers):
    unknown = set(values) - set(engram.EngramConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown Engram config fields: {sorted(unknown)}")
    config = engram.EngramConfig(**{**values, "tokenizer_name_or_path": str(tokenizer_dir)})
    for value in (config.max_ngram_size, config.n_embed_per_ngram,
                  config.n_head_per_ngram, config.kernel_size):
        if type(value) is not int or value < 1:
            raise ValueError("Engram dimensions/order/kernel must be positive integers")
    if (config.max_ngram_size < 2 or config.n_embed_per_ngram % config.n_head_per_ngram
            or len(config.engram_vocab_size) != config.max_ngram_size - 1):
        raise ValueError("inconsistent Engram order/head dimensions")
    head_offsets(config.engram_vocab_size)
    if (not config.layer_ids or len(set(config.layer_ids)) != len(config.layer_ids)
            or any(type(i) is not int or not 0 <= i < n_layers for i in config.layer_ids)):
        raise ValueError("Engram layer IDs must be distinct valid Qwen layer indices")
    return config


def _mapping(engram, config):
    names = ("engram_vocab_size", "max_ngram_size", "n_embed_per_ngram", "n_head_per_ngram",
             "layer_ids", "tokenizer_name_or_path", "pad_id", "seed")
    return engram.NgramHashMapping(**{name: getattr(config, name) for name in names})


def export_tinyengram_checkpoint(checkpoint_dir, output_dir, *, upstream_dir,
                                engram_config=None, tokenizer_dir=None, global_step=None,
                                block_rows=256, predictor="xor", shard_bytes=256 * 1024**2):
    """Convert full frozen safetensors weights without constructing a dense model.

    Non-memory tensors are copied into bounded-size shards (an individual large
    tensor may exceed the limit). Memory rows stream in blocks without dedup.
    Publish a fresh complete directory; failed conversion leaves no final bundle.
    """
    checkpoint_dir, output_dir = Path(checkpoint_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("use a fresh bundle directory")
    if type(shard_bytes) is not int or shard_bytes < 1:
        raise ValueError("shard_bytes must be a positive integer")
    keys = _checkpoint_keys(checkpoint_dir)
    backbone_config = _json(checkpoint_dir / "config.json")
    if backbone_config.get("model_type") != "qwen3":
        raise ValueError("requires a Qwen3 backbone checkpoint")
    values = engram_config if engram_config is not None else _json(checkpoint_dir / "engram_config.json")
    engram, qwen = load_upstream(upstream_dir)
    tokenizer_dir = Path(tokenizer_dir or checkpoint_dir).resolve()
    if not tokenizer_dir.is_dir():
        raise ValueError("tokenizer_dir must be a local directory")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, trust_remote_code=False)
    if global_step is None:
        trainer_state = checkpoint_dir / "trainer_state.json"
        global_step = _json(trainer_state).get("global_step") if trainer_state.exists() else None
        if global_step is None and not (values.get("warmup_steps", 0) or values.get("soft_constraint_steps", 0)):
            global_step = 0
    if type(global_step) is not int or global_step < 0:
        raise ValueError("provide a nonnegative global_step to preserve gate warmup behavior")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".meft-tiny-", dir=output_dir.parent) as temp:
        stage = Path(temp) / "bundle"
        stage.mkdir()
        tokenizer.save_pretrained(stage / "tokenizer")
        config = _config(engram, values, stage / "tokenizer", backbone_config["num_hidden_layers"])
        mapping = _mapping(engram, config)
        qwen_config = Qwen3Config.from_dict(backbone_config)
        qwen_config._attn_implementation = "eager"
        with torch.device("meta"):
            template = qwen.EngramQwenForCausalLM(qwen_config, engram_cfg=config)
        expected = template.state_dict()
        missing = set(expected) - set(keys)
        if qwen_config.tie_word_embeddings:
            missing.discard("lm_head.weight")
        if missing or set(keys) - set(expected):
            raise ValueError("requires complete backbone/gate/conv checkpoint tensors")
        for key in keys:
            if _tensor(checkpoint_dir, keys, key).shape != expected[key].shape:
                raise ValueError(f"checkpoint tensor shape differs from config: {key}")
        del template, expected
        layers, excluded = {}, set()
        for layer_id in config.layer_ids:
            prefix = f"model.engram_layers.{layer_id}.multi_head_embedding."
            weight_key, offsets_key = prefix + "embedding.weight", prefix + "offsets"
            sizes = [n for group in mapping.vocab_size_across_layers[layer_id] for n in group]
            if not torch.equal(_tensor(checkpoint_dir, keys, offsets_key), torch.tensor(head_offsets(sizes))):
                raise ValueError(f"checkpoint head offsets differ: layer {layer_id}")
            metadata = export_multihead(
                _tensor(checkpoint_dir, keys, weight_key), sizes, stage / "memory" / str(layer_id),
                block_rows=block_rows, backend="native", deduplicate=False, predictor=predictor,
            )
            if metadata["shape"][1] != config.n_embed_per_ngram // config.n_head_per_ngram:
                raise ValueError("checkpoint memory dimension differs from config")
            layers[str(layer_id)] = {"path": f"memory/{layer_id}", "head_sizes": sizes,
                                     "weight_key": weight_key, "offsets_key": offsets_key,
                                     "weight_sha256": metadata["weight_sha256"]}
            excluded.update((weight_key, offsets_key))
        declared_prefixes = {f"model.engram_layers.{i}." for i in config.layer_ids}
        if any(key.startswith("model.engram_layers.") and
               not any(key.startswith(p) for p in declared_prefixes) for key in keys):
            raise ValueError("checkpoint contains undeclared Engram layers")
        # Keep each original backbone/gate/conv tensor unchanged. No state is
        # dropped based on a token/query trace; only mapped tables are removed.
        weight_map, shard, size, shard_index = {}, {}, 0, 0

        def flush():
            nonlocal shard, size, shard_index
            if shard:
                shard_index += 1
                name = f"backbone-{shard_index:05d}.safetensors"
                save_file(shard, stage / name)
                weight_map.update({key: name for key in shard})
                shard, size = {}, 0

        for key in sorted(set(keys) - excluded):
            tensor = _tensor(checkpoint_dir, keys, key)
            nbytes = tensor.numel() * tensor.element_size()
            if size + nbytes > shard_bytes:
                flush()
            shard[key], size = tensor, size + nbytes
        flush()
        _write_json(stage / "config.json", backbone_config)
        if (checkpoint_dir / "generation_config.json").exists():
            shutil.copyfile(checkpoint_dir / "generation_config.json", stage / "generation_config.json")
        config_dict = asdict(config)
        config_dict["tokenizer_name_or_path"] = "tokenizer"
        _write_json(stage / "engram_config.json", config_dict)
        _write_json(stage / "model.safetensors.index.json", {"weight_map": weight_map})
        files = {str(path.relative_to(stage)): file_sha256(path) for path in sorted(stage.rglob("*")) if path.is_file()}
        manifest = {"schema_version": 1, "upstream_sha": UPSTREAM_SHA, "source_hashes": SOURCE_HASHES,
                    "transformers_version": "4.57.3", "layers": layers, "global_step": global_step,
                    "token_lookup_sha256": _hash_lookup(mapping), "files": files,
                    "source_checkpoint_files": {name: file_sha256(_safe_path(checkpoint_dir, name))
                                                for name in sorted(set(keys.values()))},
                    "contract": "frozen CPU memory; original hash IDs and weight bytes; complete Qwen3 backbone"}
        _write_json(stage / "bundle.json", manifest)
        stage.rename(output_dir)
    return manifest


@dataclass
class TinyEngramCache:
    """One request's KV cache, raw-token tail and per-layer convolution tails."""
    kv: DynamicCache
    owner: str
    seen_tokens: int = 0
    token_tail: torch.Tensor | None = None
    conv_tails: dict = field(default_factory=dict)
    valid: bool = True


@dataclass
class _ForwardContext:
    cache: TinyEngramCache | None
    host_ids: torch.Tensor
    hashes: dict = field(default_factory=dict)
    prefetched: dict = field(default_factory=dict)


class _HistoryHash:
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def hash(self, input_ids):
        context = _decode_context.get()
        if context is None:
            return self.original.hash(input_ids)
        key = id(self.original)
        if key not in context.hashes:
            cache, current_ids = context.cache, context.host_ids
            all_ids = (current_ids if cache is None or cache.token_tail is None
                       else torch.cat((cache.token_tail, current_ids), dim=1))
            context.hashes[key] = {k: v[:, -current_ids.shape[1]:]
                                   for k, v in self.original.hash(all_ids.numpy()).items()}
        return context.hashes[key]


class TinyEngramModel(nn.Module):
    """Inference wrapper, with CPU raw IDs and request-owned device KV/conv state.

    Adds missing n-gram/ShortConv history to the upstream incremental path.
    No-cache forward still calls upstream hashing/convolution unchanged.
    It does not implement the Hugging Face generation/beam/padding API.
    """
    def __init__(self, model):
        super().__init__()
        self.model, self.config = model, model.config
        self.device = next(model.parameters()).device
        self._cache_owner = uuid.uuid4().hex
        for layer_id, layer in model.model.engram_layers.items():
            if isinstance(layer.hash_mapping, _HistoryHash):
                raise ValueError("model is already wrapped")
            layer.hash_mapping = _HistoryHash(layer.hash_mapping)
            if isinstance(layer.multi_head_embedding, StagedMultiHeadEmbedding):
                embedding = layer.multi_head_embedding
                original_layer = layer.forward
                original_gather = embedding.forward

                def host_layer(hidden_states, input_ids, *, original=original_layer):
                    context = _decode_context.get()
                    if context is None:
                        raise RuntimeError("staged Engram layers must run through TinyEngramModel")
                    # Upstream then hashes CPU IDs and avoids its GPU->CPU->GPU
                    # hash-ID round trip. Hidden states/gates remain on device.
                    return original(hidden_states, context.host_ids)

                def prefetched_gather(ids, *, original=original_gather, layer_id=layer_id):
                    context = _decode_context.get()
                    if context is None or layer_id not in context.prefetched:
                        return original(ids)
                    expected_ids, ticket = context.prefetched[layer_id]
                    if not torch.equal(ids, expected_ids):
                        raise RuntimeError("prefetched hash IDs differ from the layer query")
                    return ticket.wait()

                layer.forward = host_layer
                embedding.forward = prefetched_gather
            original = layer.short_conv.forward
            keep = (layer.short_conv.conv.kernel_size[0] - 1) * layer.short_conv.conv.dilation[0]

            def cached_conv(x, *, original=original, keep=keep, layer_id=layer_id):
                context = _decode_context.get()
                if context is None or context.cache is None:
                    return original(x)
                cache = context.cache
                tail = cache.conv_tails.get(layer_id)
                full = x if tail is None else torch.cat((tail, x), dim=1)
                result = original(full)[:, -x.shape[1]:]
                cache.conv_tails[layer_id] = full[:, -keep:].detach().clone() if keep else full[:, :0]
                return result

            layer.short_conv.forward = cached_conv

    def forward(self, input_ids, *, past_key_values=None, use_cache=False, attention_mask=None, **kwargs):
        if (input_ids.device.type != "cpu" or input_ids.ndim != 2 or not all(input_ids.shape)
                or input_ids.dtype not in (torch.int32, torch.int64)):
            raise ValueError("requires a nonempty 2-D CPU integer token batch")
        if any(p.device != self.device for p in self.model.parameters()):
            raise ValueError("model placement changed; load a new bundle for the target device")
        staged = [layer for layer in self.model.model.engram_layers.values()
                  if isinstance(layer.multi_head_embedding, StagedMultiHeadEmbedding)]
        if self.device.type != "cpu" and any(isinstance(layer.multi_head_embedding, MappedMultiHeadEmbedding)
                                            for layer in self.model.model.engram_layers.values()):
            raise ValueError("accelerator models require staged host memory")
        if self.training and use_cache:
            raise ValueError("incremental cache is inference only; call eval()")
        if any(name in kwargs for name in ("inputs_embeds", "position_ids", "cache_position")):
            raise ValueError("custom embeddings/positions are not supported")
        if kwargs.get("return_dict", True) is not True:
            raise ValueError("return_dict=False is not supported")
        cache = past_key_values
        if not use_cache and cache is not None:
            raise ValueError("past_key_values requires use_cache=True")
        if use_cache:
            if cache is None:
                cache = TinyEngramCache(DynamicCache(config=self.config), self._cache_owner)
            if not isinstance(cache, TinyEngramCache) or not cache.valid:
                raise ValueError("requires a valid TinyEngramCache returned by this wrapper")
            if cache.owner != self._cache_owner:
                raise ValueError("cache belongs to a different model")
            if cache.kv.get_seq_length() != cache.seen_tokens:
                raise ValueError("KV cache length differs from memory history")
            if cache.token_tail is not None and cache.token_tail.shape[0] != input_ids.shape[0]:
                raise ValueError("cached batch size cannot change")
        seen = cache.seen_tokens if cache is not None else 0
        if attention_mask is not None and (attention_mask.shape != (input_ids.shape[0], seen + input_ids.shape[1])
                                           or not bool((attention_mask == 1).all())):
            raise ValueError("only unpadded all-one attention masks are supported")
        context = _ForwardContext(cache, input_ids) if use_cache or staged else None
        token = _decode_context.set(context)
        if cache is not None:
            cache.valid = False  # KV/conv updates cannot be rolled back on a failed forward.
        try:
            # Hashing needs no hidden states, so queue every memory layer before
            # the first decoder block. H2D can overlap the intervening backbone.
            for layer_id, layer in self.model.model.engram_layers.items():
                if isinstance(layer.multi_head_embedding, StagedMultiHeadEmbedding):
                    ids = torch.from_numpy(layer.hash_mapping.hash(input_ids.numpy())[int(layer_id)])
                    context.prefetched[layer_id] = (ids, layer.multi_head_embedding.prefetch(ids))
            model_kwargs = dict(kwargs)
            if isinstance(model_kwargs.get("labels"), torch.Tensor):
                model_kwargs["labels"] = model_kwargs["labels"].to(self.device)
            output = self.model(input_ids=input_ids.to(self.device),
                                attention_mask=attention_mask.to(self.device) if attention_mask is not None else None,
                                past_key_values=cache.kv if cache else None, use_cache=use_cache, **model_kwargs)
            if cache is not None:
                tail = input_ids if cache.token_tail is None else torch.cat((cache.token_tail, input_ids), dim=1)
                keep = self.model.model.engram_cfg.max_ngram_size - 1
                cache.token_tail = tail[:, -keep:].detach().clone()
                cache.seen_tokens += input_ids.shape[1]
                cache.valid = True
                output.past_key_values = cache
            return output
        finally:
            _decode_context.reset(token)

    @torch.inference_mode()
    def generate(self, input_ids, *, max_new_tokens, use_cache=True):
        """Fixed-length greedy decode; no EOS stopping, sampling, beams or padding."""
        if self.training or type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("call eval() and provide a nonnegative integer max_new_tokens")
        sequence, cache = input_ids.clone(), None
        for _ in range(max_new_tokens):
            current = sequence if cache is None else sequence[:, -1:]
            output = self(current, past_key_values=cache, use_cache=use_cache)
            next_ids = output.logits[:, -1].argmax(-1, keepdim=True).cpu()
            sequence = torch.cat((sequence, next_ids), dim=1)
            cache = output.past_key_values if use_cache else None
        return sequence

    def close(self):
        for module in self.model.modules():
            if isinstance(module, (StagedMultiHeadEmbedding, MappedMultiHeadEmbedding)):
                module.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def load_tinyengram_checkpoint(output_dir, *, upstream_dir, native_library=None,
                              native_cache_bytes=0, query_chunk_rows=65536, device="cpu",
                              prefetch_memory=False, staging_rows=65536, staging_slots=2,
                              native_cache_policy="blocks"):
    """Construct on meta, assign original tensors, optionally place the backbone.

    Does not allocate original dense memory tables. File checksums scan the whole
    bundle; this is not lazy startup. The original full checkpoint is unnecessary.
    device=cuda/mps keeps tables/IDs on CPU and prefetches rows before decoder
    execution. CPU can use the same scheduler with prefetch_memory=True. CUDA
    overlap and performance need real hardware validation; MPS is synchronous.
    """
    target = _device(device)
    root = Path(output_dir).resolve()
    manifest = _json(root / "bundle.json")
    if (manifest["schema_version"] != 1 or manifest["upstream_sha"] != UPSTREAM_SHA
            or manifest["source_hashes"] != SOURCE_HASHES):
        raise ValueError("unsupported TinyEngram bundle provenance")
    for name, digest in manifest["files"].items():
        if file_sha256(_safe_path(root, name)) != digest:
            raise ValueError(f"bundle checksum mismatch: {name}")
    engram, qwen = load_upstream(upstream_dir)
    config = Qwen3Config.from_dict(_json(root / "config.json"))
    config._attn_implementation = "eager"
    engram_config = _config(engram, _json(root / "engram_config.json"), root / "tokenizer", config.num_hidden_layers)
    engram.set_global_step(manifest["global_step"])
    with torch.device("meta"):
        model = qwen.EngramQwenForCausalLM(config, engram_cfg=engram_config)
    # Rotary frequencies are nonpersistent buffers, absent from state_dict.
    # Recreate them on CPU rather than leaving the meta constructor's buffers.
    model.model.rotary_emb = type(model.model.rotary_emb)(config=config, device="cpu")
    opened = []
    try:
        if set(manifest["layers"]) != {str(i) for i in engram_config.layer_ids}:
            raise ValueError("bundle layers differ from Engram config")
        for layer_id, layer in model.model.engram_layers.items():
            entry = manifest["layers"][layer_id]
            sizes = [n for group in layer.hash_mapping.vocab_size_across_layers[int(layer_id)] for n in group]
            mapped = MappedMultiHeadEmbedding(_safe_path(root, entry["path"]), native_library=native_library,
                                             native_cache_bytes=native_cache_bytes,
                                             native_cache_policy=native_cache_policy,
                                             query_chunk_rows=query_chunk_rows)
            opened.append(mapped)
            if (sizes != entry["head_sizes"] or sizes != mapped.metadata["head_sizes"]
                    or mapped.metadata["weight_sha256"] != entry["weight_sha256"]
                    or mapped.embedding_dim != engram_config.n_embed_per_ngram // engram_config.n_head_per_ngram
                    or _hash_lookup(layer.hash_mapping) != manifest["token_lookup_sha256"]):
                raise ValueError("mapped memory/hash layout differs from bundle")
            layer.multi_head_embedding = mapped
        keys = _checkpoint_keys(root)
        expected = model.state_dict()
        # Mapped offsets/head sizes are already initialized. A tied output head
        # may be absent from ordinary HF safetensors; restore it via tie_weights.
        mapped_keys = {f"model.engram_layers.{i}.multi_head_embedding.{name}"
                       for i in engram_config.layer_ids for name in ("offsets", "head_sizes")}
        allowed_missing = mapped_keys | ({"lm_head.weight"} if config.tie_word_embeddings else set())
        if set(keys) - set(expected) or set(expected) - set(keys) - allowed_missing:
            raise ValueError("checkpoint has missing/unexpected backbone or Engram tensors")
        for filename in sorted(set(keys.values())):
            with safe_open(_safe_path(root, filename), framework="pt", device="cpu") as handle:
                state = {key: handle.get_tensor(key) for key in handle.keys()}
            model.load_state_dict(state, strict=False, assign=True)
        model.tie_weights()
        if any(t.device.type == "meta" for t in (*model.parameters(), *model.buffers())):
            raise ValueError("checkpoint left uninitialized model tensors")
        if prefetch_memory or target.type != "cpu":
            for layer_id, layer in model.model.engram_layers.items():
                old = layer.multi_head_embedding
                staged = StagedMultiHeadEmbedding(
                    old._output_dir, device=target, native_library=native_library,
                    native_cache_bytes=native_cache_bytes, query_chunk_rows=query_chunk_rows,
                    staging_rows=staging_rows, staging_slots=staging_slots, native_cache_policy=native_cache_policy,
                )
                opened.append(staged)
                layer.multi_head_embedding = staged
                old.close()
            model.to(target)
        return TinyEngramModel(model.requires_grad_(False)).eval()
    except BaseException:
        for mapped in opened:
            mapped.close()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--upstream-dir", required=True)
    parser.add_argument("--engram-config")
    parser.add_argument("--tokenizer-dir")
    parser.add_argument("--global-step", type=int)
    parser.add_argument("--block-rows", type=int, default=256)
    parser.add_argument("--predictor", choices=("none", "xor"), default="xor")
    args = parser.parse_args()
    manifest = export_tinyengram_checkpoint(
        args.checkpoint, args.output_dir, upstream_dir=args.upstream_dir,
        engram_config=_json(args.engram_config) if args.engram_config else None,
        tokenizer_dir=args.tokenizer_dir, global_step=args.global_step,
        block_rows=args.block_rows, predictor=args.predictor,
    )
    print(json.dumps({"output_dir": str(Path(args.output_dir).resolve()), "layers": list(manifest["layers"])}, indent=2))


if __name__ == "__main__":
    main()
