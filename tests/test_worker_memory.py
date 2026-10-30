"""Mapped adapters copy/share configuration while each worker owns its handle."""
import copy
import multiprocessing
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None


def _read_in_spawned_worker(module, queue):
    try:
        result = module(torch.tensor([[2, 3], [0, 0]]))
        queue.put((os.getpid(), result.view(torch.uint8).numpy().tobytes()))
    except Exception as error:
        queue.put(("error", repr(error)))
    finally:
        module.close()


@unittest.skipUnless(torch is not None and shutil.which("cc"), "PyTorch and compiler required")
class WorkerMemoryTest(unittest.TestCase):
    def test_query_budget_preserves_multihead_factory_and_copy_results(self):
        from tools.engram_row_adapter import MappedMultiHeadEmbedding, export_multihead, open_demo_embedding
        with tempfile.TemporaryDirectory() as tmp:
            weight = torch.arange(21).float().reshape(7, 3)
            export_multihead(weight, [3, 4], tmp, backend="native", block_rows=2)
            query = torch.tensor([[[2, 3], [0, 0]], [[0, 1], [2, 3]]]).transpose(0, 1)
            expected = torch.nn.functional.embedding(query + torch.tensor([0, 3]), weight)
            with MappedMultiHeadEmbedding(tmp, native_library=self.library, query_chunk_rows=1) as original:
                self.assertTrue(torch.equal(original(query), expected))
                self.assertEqual(original.table.last_block_reads, query.numel())
                with copy.deepcopy(original) as clone:
                    self.assertEqual(clone.query_chunk_rows, 1)
                    self.assertTrue(torch.equal(clone(query), expected))
                state = original.__getstate__()
                state.pop("query_chunk_rows")  # Previously serialized module settings.
                restored = MappedMultiHeadEmbedding.__new__(MappedMultiHeadEmbedding)
                restored.__setstate__(state)
                with restored:
                    self.assertEqual(restored.query_chunk_rows, 65536)
                    self.assertTrue(torch.equal(restored(query), expected))
            with open_demo_embedding(tmp, [3, 4], 3, native_library=self.library, query_chunk_rows=2) as factory:
                self.assertTrue(torch.equal(factory(query), expected))
            for invalid in (0, -1, True, 1.5):
                with self.assertRaises(ValueError):
                    MappedMultiHeadEmbedding(tmp, native_library=self.library, query_chunk_rows=invalid)

    def test_ordinary_embedding_and_unique_bag_use_the_query_budget(self):
        from tools.engram_row_adapter import MappedEmbedding, export_embedding
        from tools.mapped_embedding_bag import export_embedding_bag, open_embedding_bag
        with tempfile.TemporaryDirectory() as tmp:
            source = torch.nn.Embedding(8, 4)
            export_embedding(source, Path(tmp) / "embedding", block_rows=2)
            ids = torch.tensor([[7, 0, 7, 1]])
            with MappedEmbedding(Path(tmp) / "embedding", native_library=self.library, query_chunk_rows=1) as mapped:
                self.assertTrue(torch.equal(mapped(ids), source(ids)))
                self.assertEqual(mapped.gather.table.last_block_reads, ids.numel())
            bag = torch.nn.EmbeddingBag(8, 4, mode="sum")
            export_embedding_bag(bag, Path(tmp) / "bag", block_rows=2)
            with open_embedding_bag(Path(tmp) / "bag", native_library=self.library, query_chunk_rows=1) as mapped:
                self.assertTrue(torch.equal(mapped(ids), bag(ids)))
                self.assertEqual(mapped.last_unique_rows, 3)
                self.assertEqual(mapped.gather.table.last_block_reads, 3)

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.library = Path(cls.temp.name) / "rows.so"
        subprocess.run(["cc", "-std=c11", "-O2", "-fPIC",
                        "-dynamiclib" if sys.platform == "darwin" else "-shared",
                        str(Path(__file__).parents[1] / "native/meft_rows.c"),
                        "-lz", "-o", str(cls.library)], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_deepcopy_and_pickle_keep_exact_bytes_and_independent_handles(self):
        from tools.engram_row_adapter import MappedMultiHeadEmbedding, export_multihead
        for backend in ("python", "native"):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                with self.subTest(backend=backend, dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                    rows = torch.arange(21).to(dtype).reshape(7, 3)
                    rows[2, 0] = -0.
                    export_multihead(rows, [3, 4], tmp, block_rows=2, backend=backend)
                    with MappedMultiHeadEmbedding(tmp, cache_blocks=1, native_library=self.library,
                                                 native_cache_bytes=48 if backend == "native" else 0,
                                                 native_cache_policy="rows" if backend == "native" else "blocks") as original:
                        query = torch.tensor([[2, 3], [0, 0]])
                        expected = original(query).view(torch.uint8).clone()
                        for clone in (copy.copy(original), copy.deepcopy(original), pickle.loads(pickle.dumps(original))):
                            with clone:
                                self.assertIsNot(clone.table, original.table)
                                if backend == "native":
                                    self.assertNotEqual(clone.table.handle, original.table.handle)
                                    self.assertEqual(clone.table.cache_bytes, 0)
                                    self.assertEqual(clone.table.cache_policy, "rows")
                                else:
                                    self.assertIsNot(clone.table.mapping, original.table.mapping)
                                    self.assertFalse(clone.table.cache)
                                self.assertTrue(torch.equal(clone(query).view(torch.uint8), expected))
                        # Closing every clone must not invalidate its source.
                        self.assertTrue(torch.equal(original(query).view(torch.uint8), expected))
                    with self.assertRaisesRegex(ValueError, "closed"):
                        pickle.dumps(original)
                    with self.assertRaisesRegex(ValueError, "closed"):
                        original(torch.empty((0, 2), dtype=torch.int64))

    def test_spawn_uses_separate_handles_and_captured_paths(self):
        from tools.engram_row_adapter import MappedMultiHeadEmbedding, export_multihead
        context = multiprocessing.get_context("spawn")
        for backend in ("python", "native"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp:
                export_multihead(torch.arange(21).float().reshape(7, 3), [3, 4], tmp, backend=backend)
                with MappedMultiHeadEmbedding(tmp, native_library=self.library) as original:
                    expected = original(torch.tensor([[2, 3], [0, 0]])).view(torch.uint8).numpy().tobytes()
                    queue = context.Queue()
                    process = context.Process(target=_read_in_spawned_worker, args=(original, queue))
                    try:
                        process.start()
                        pid, output = queue.get(timeout=30)
                        process.join(timeout=10)
                        self.assertEqual(process.exitcode, 0)
                        self.assertNotEqual(pid, os.getpid())
                        self.assertEqual(output, expected)
                        self.assertEqual(original(torch.tensor([[2, 3], [0, 0]])).view(torch.uint8).numpy().tobytes(), expected)
                    finally:
                        if process.is_alive():
                            process.terminate()
                            process.join(timeout=5)
                        queue.close()
                        queue.join_thread()

    def test_serialized_state_contains_no_table_payload_and_rejects_changed_artifact(self):
        from tools.engram_row_adapter import MappedMultiHeadEmbedding, export_multihead
        for backend in ("python", "native"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp:
                export_multihead(torch.arange(65536).float().reshape(4096, 16), [4096], tmp, backend=backend)
                with MappedMultiHeadEmbedding(tmp, native_library=self.library) as original:
                    state = pickle.dumps(original)
                    self.assertLess(len(state), 16384)
                    self.assertNotIn("table", original.__getstate__())
                    table = Path(tmp) / original.metadata["table_file"]
                    payload = bytearray(table.read_bytes())
                    payload[-1] ^= 1
                    table.write_bytes(payload)
                    with self.assertRaisesRegex(ValueError, "checksum"):
                        pickle.loads(state)

    def test_restore_keeps_environment_selected_library_and_relative_artifact_path(self):
        from unittest.mock import patch
        from tools.engram_row_adapter import MappedMultiHeadEmbedding, export_multihead
        previous_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "artifact"
            export_multihead(torch.arange(21).float().reshape(7, 3), [3, 4], directory, backend="native")
            with patch.dict(os.environ, {"MEFT_ROWS_LIBRARY": str(self.library)}):
                original = MappedMultiHeadEmbedding(os.path.relpath(directory))
            try:
                state = pickle.dumps(original)
                os.chdir(tmp)
                with patch.dict(os.environ, {"MEFT_ROWS_LIBRARY": "/missing/rows.so"}):
                    with pickle.loads(state) as clone:
                        self.assertTrue(torch.equal(clone(torch.tensor([[2, 3]])), original(torch.tensor([[2, 3]]))))
            finally:
                os.chdir(previous_cwd)
                original.close()

    def test_wrapped_embedding_and_weighted_bag_survive_copy(self):
        from tools.engram_row_adapter import export_embedding, MappedEmbedding
        from tools.mapped_embedding_bag import export_embedding_bag, open_embedding_bag
        with tempfile.TemporaryDirectory() as tmp:
            source = torch.nn.Embedding(8, 4)
            embedding_dir = Path(tmp) / "embedding"
            export_embedding(source, embedding_dir)
            with MappedEmbedding(embedding_dir, native_library=self.library) as original:
                with copy.deepcopy(original) as clone:
                    self.assertTrue(torch.equal(clone(torch.tensor([7, 0, 7])), source(torch.tensor([7, 0, 7]))))
            bag = torch.nn.EmbeddingBag(8, 4, mode="sum")
            bag_dir = Path(tmp) / "bag"
            export_embedding_bag(bag, bag_dir)
            for strategy in ("unique", "chunked"):
                with open_embedding_bag(bag_dir, native_library=self.library, strategy=strategy) as original:
                    with copy.deepcopy(original) as clone:
                        ids, weights = torch.tensor([[7, 0, 7]]), torch.tensor([[.5, -1., 2.]])
                        self.assertTrue(torch.equal(clone(ids, per_sample_weights=weights), original(ids, per_sample_weights=weights)))


if __name__ == "__main__":
    unittest.main()
