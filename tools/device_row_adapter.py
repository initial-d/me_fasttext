"""Host-resident native memory, bounded row staging and explicit CUDA prefetch.

CPU hashing supplies head-local IDs. Only queried vectors cross to the target
device. CUDA copies are asynchronous; CPU and MPS are synchronous references.
Each query owns its output; only host staging buffers are pooled. No training,
distributed scheduling, CUDA graph capture or background gather threads.
"""
from dataclasses import dataclass
import threading

import torch
from torch import nn

from .engram_row_adapter import MappedMultiHeadEmbedding


def _device(device):
    device = torch.device(device)
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError("target device must be cpu, cuda or mps")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; this is not a GPU validation")
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
    elif device.type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")
        device = torch.device("mps", 0)
    else:
        device = torch.device("cpu")
    return device


class GatherTicket:
    """One owned gather output; wait before consumption on each CUDA stream.

    wait() enqueues a consumer-stream event dependency, without blocking the
    host. It also records allocator ownership for that consumer. Repeated waits
    and waits after adapter.close() are safe. The ticket need not be retained
    once its returned tensor is used on the stream passed to wait().
    """
    def __init__(self, output, event=None):
        self._output, self._event = output, event

    @property
    def ready(self):
        return self._event is None or self._event.query()

    def wait(self, stream=None):
        if self._event is None:
            if stream is not None:
                raise ValueError("streams are only supported for CUDA outputs")
            return self._output
        stream = stream or torch.cuda.current_stream(self._output.device)
        if stream.device != self._output.device:
            raise ValueError("consumer stream belongs to a different CUDA device")
        stream.wait_event(self._event)
        self._output.record_stream(stream)
        return self._output


@dataclass
class _Slot:
    tensor: torch.Tensor | None = None
    event: object = None


class StagedMultiHeadEmbedding(nn.Module):
    """Immutable native table on CPU, queried vectors on a fixed target device.

    Input IDs must be CPU int32/int64 [..., heads]. Output keeps original dtype
    and shape [..., heads, embedding_dim]. prefetch() gathers on the caller's CPU
    thread, then queues CUDA H2D on a dedicated copy stream. It does not promise
    asynchronous disk/decompression. More chunks than staging slots can block
    the CPU until an earlier copy releases its slot. Total host staging is at
    most staging_slots * staging_rows * row_bytes; output, IDs, native caches,
    mmap/RSS and model tensors are separate allocations.

    A single worker owns this module. Concurrent gather/close is rejected.
    Parent .to(device) keeps native metadata on CPU; changing the configured
    target or stored dtype requires constructing a new adapter.
    """
    def __init__(self, output_dir, *, device="cuda", native_library=None,
                 native_cache_bytes=0, query_chunk_rows=65536,
                 staging_rows=65536, staging_slots=2):
        super().__init__()
        for name, value in (("staging_rows", staging_rows), ("staging_slots", staging_slots)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.device = _device(device)
        self.gather = MappedMultiHeadEmbedding(
            output_dir, native_library=native_library, native_cache_bytes=native_cache_bytes,
            query_chunk_rows=query_chunk_rows,
        )
        if self.gather.backend != "native":
            self.gather.close()
            raise ValueError("staged device gather requires a native table")
        self.num_heads, self.embedding_dim = self.gather.num_heads, self.gather.embedding_dim
        self.dtype = self.gather.dtype
        self.staging_rows, self.staging_slots = staging_rows, staging_slots
        self._slots = [_Slot() for _ in range(staging_slots)]
        self._next_slot, self._closed = 0, False
        self._lock = threading.Lock()
        self._copy_stream = None
        self._stats = {"queries": 0, "copy_chunks": 0, "slot_allocations": 0,
                       "slot_reuses": 0, "reuse_waits": 0}
        try:
            if self.device.type == "cuda":
                self._copy_stream = torch.cuda.Stream(device=self.device)
        except BaseException:
            self.gather.close()
            raise
        self.training = False

    @property
    def stats(self):
        return {**self._stats, "staging_bytes": sum(
            s.tensor.numel() * s.tensor.element_size() for s in self._slots if s.tensor is not None),
            "staging_byte_limit": self.staging_slots * self.staging_rows * self.embedding_dim *
            torch.empty((), dtype=self.dtype).element_size(),
            "pinned": self.device.type == "cuda"}

    def train(self, mode=True):
        if mode:
            raise RuntimeError("staged memory is immutable and supports inference only")
        return super().train(False)

    def _apply(self, fn, recurse=True):
        # nn.Module.to() recurses here while placing the surrounding backbone.
        # Never move the table's offsets or silently convert its stored bytes.
        probe = fn(torch.empty(0, device=self.device, dtype=self.dtype))
        if probe.device != self.device or probe.dtype != self.dtype:
            raise ValueError("staged memory target/dtype is fixed; construct a new adapter")
        return self

    def __getstate__(self):
        raise TypeError("live staged adapters cannot be serialized; reopen from the bundle in each worker")

    def _slot(self):
        slot = self._slots[self._next_slot]
        self._next_slot = (self._next_slot + 1) % self.staging_slots
        if slot.tensor is None:
            slot.tensor = torch.empty((self.staging_rows, self.embedding_dim), dtype=self.dtype,
                                      device="cpu", pin_memory=self.device.type == "cuda")
            self._stats["slot_allocations"] += 1
        else:
            self._stats["slot_reuses"] += 1
        # CUDA cannot race with the C reader's direct writes into pinned memory.
        if slot.event is not None and not slot.event.query():
            self._stats["reuse_waits"] += 1
            slot.event.synchronize()
        return slot

    def _read(self, ids, output):
        self.gather.table.gather_into(
            ids.data_ptr(), ids.numel(), output.data_ptr(), output.numel() * output.element_size(),
            max_query_rows=self.gather.query_chunk_rows,
        )

    @torch.no_grad()
    def prefetch(self, input_ids):
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("native CPU gather is not supported inside CUDA graph capture")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("staged memory is single-worker; concurrent gather/close is unsupported")
        try:
            if self._closed:
                raise ValueError("staged memory is closed")
            ids = self.gather._global_ids(input_ids).reshape(-1)
            shape = (*input_ids.shape, self.embedding_dim)
            if self.device.type == "cpu":
                output = torch.empty(shape, dtype=self.dtype, device="cpu")
                self._read(ids, output)
                self._stats["queries"] += 1
                return GatherTicket(output)
            # Allocate each result on its producing stream. Later wait() calls
            # protect both data readiness and CUDA allocator stream ownership.
            context = (torch.cuda.stream(self._copy_stream) if self._copy_stream is not None
                       else torch.device(self.device))
            with context:
                output = torch.empty(shape, dtype=self.dtype, device=self.device)
                flat = output.reshape(-1, self.embedding_dim)
                if ids.numel() == 0:
                    self._read(ids, torch.empty(0, dtype=self.dtype, device="cpu"))
                for start in range(0, ids.numel(), self.staging_rows):
                    end = min(start + self.staging_rows, ids.numel())
                    slot = self._slot()
                    staging = slot.tensor[:end - start]
                    self._read(ids[start:end], staging)
                    flat[start:end].copy_(staging, non_blocking=self.device.type == "cuda")
                    if self._copy_stream is not None:
                        slot.event = torch.cuda.Event()
                        slot.event.record(self._copy_stream)
                    self._stats["copy_chunks"] += 1
                done = None
                if self._copy_stream is not None:
                    done = torch.cuda.Event()
                    done.record(self._copy_stream)
                self._stats["queries"] += 1
                return GatherTicket(output, done)
        except BaseException:
            # A failed copy/event creation must not release a host slot early.
            if self._copy_stream is not None:
                self._copy_stream.synchronize()
            raise
        finally:
            self._lock.release()

    def forward(self, input_ids):
        return self.prefetch(input_ids).wait()

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("staged memory is single-worker; concurrent gather/close is unsupported")
        try:
            if not self._closed:
                if self._copy_stream is not None:
                    self._copy_stream.synchronize()
                self.gather.close()
                self._slots.clear()
                self._closed = True
        finally:
            self._lock.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
