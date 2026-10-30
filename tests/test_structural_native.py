"""Real bundled DATrie tests; synthetic rows are correctness fixtures only."""
import math
import os
from pathlib import Path
import platform
import struct
import subprocess
import tempfile
import unittest

from tools.native_row_store import NativeRows, pack_native
from tools.structural_native import StructuralPlan, StructuralIndex

ROOT = Path(__file__).resolve().parents[1]


class CompactionTest(unittest.TestCase):
    def test_rejects_dangling_chained_out_of_range_and_noninteger_maps(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "raw", Path(tmp) / "men"
            source.write_bytes(b"abcd")
            for mapping in ([0], [0, 4, 2, 3], [0, 0, 1, 3], [0, 2, 1, 3], [False, 1, 2, 3]):
                with self.assertRaisesRegex(ValueError, "idempotent"):
                    pack_native(source, output, 1, representatives=mapping)
                self.assertFalse(output.exists())


@unittest.skipUnless(platform.system() == "Linux" and platform.machine() in ("x86_64", "AMD64"),
                     "actual bundled DATrie archives target Linux x86-64")
class RealStructuralTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.library = Path(cls.tmp.name) / "libstructural.so"
        cls.rows_library = Path(cls.tmp.name) / "librows.so"
        subprocess.run([os.environ.get("CXX", "c++"), "-std=c++11", "-O2", "-Wall", "-Wextra", "-Werror",
            "-fPIC", "-shared", "-I" + str(ROOT / "src"), "-I" + str(ROOT / "mylib/include"),
            str(ROOT / "native/meft_structural.cc"), "-L" + str(ROOT / "mylib/lib"), "-ltrie", "-lpub",
            "-o", str(cls.library)], check=True)
        subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-O2", "-fPIC", "-shared",
            str(ROOT / "native/meft_rows.c"), "-lz", "-o", str(cls.rows_library)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_prefix_suffix_utf8_keys_original_ids_compact_forwarding_and_both_caches(self):
        keys = {1: "a", 2: "ab", 3: "xab", 4: "记", 5: "记忆", 6: "\na", 7: "z"}
        rows = [(0., 0.), (1., 0.), (1., 0.), (1., 0.), (0., 1.), (0., 1.), (1., 0.), (-1., 0.)]
        raw = b"".join(struct.pack("<2f", *r) for r in rows)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "raw"
            source.write_bytes(raw)
            with StructuralPlan(raw, 2, keys, self.library, mode="exact") as plan:
                reps = plan.representatives
                self.assertEqual(reps[0], 0)
                self.assertEqual(reps[2], 1)
                self.assertEqual(reps[5], 4)
                self.assertEqual(reps[7], 7)
                self.assertTrue(all(reps[r] == r for r in reps))
                self.assertLess(len(set(reps)), len(rows))
                for index in ("flat", "datrie", "hash"):
                    self.assertEqual(plan.lookup([keys[i] for i in keys] + ["unknown", ""], index=index),
                                     list(keys) + [None, None])
                metadata = plan.export(source, Path(tmp) / "compact", block_rows=2)
                self.assertTrue(metadata["original_bytes_preserved"])
                self.assertEqual(metadata["unique_rows"], len(set(reps)))
                self.assertEqual(metadata["marked_rows"], len(set(reps)))
                # Dense physical IDs preserve all ORIGINAL numeric IDs.
                for policy in ("rows", "blocks"):
                    with NativeRows(Path(tmp) / "compact/table.men", self.rows_library,
                                    cache_bytes=1024, cache_policy=policy) as table:
                        for _ in range(2):
                            self.assertEqual(table.rows([5, 0, 2, 1, 6, 3, 7, 5]),
                                b"".join(raw[i*8:(i+1)*8] for i in [5, 0, 2, 1, 6, 3, 7, 5]))
                with StructuralIndex(Path(tmp) / "compact/keys.das", self.library) as index:
                    self.assertEqual(index.lookup(["记忆", "a", "xab", "unknown"]), [5, 1, 3, None])
                    with self.assertRaisesRegex(ValueError, "arguments"):
                        index.lookup(["a"], index="flat")

    def test_joint_final_guard_rejects_direction_norm_and_chain_drift(self):
        keys = {i: key for i, key in enumerate(["a", "ab", "abc", "ax", "ay", "az"]) }
        rows = [(1., 0.), (.996, .089), (.984, .178), (-1., 0.), (2., 0.), (math.nan, 0.)]
        raw = b"".join(struct.pack("<2f", *r) for r in rows)
        with StructuralPlan(raw, 2, keys, self.library, mode="joint") as plan:
            self.assertEqual(plan.representatives[1], 0)
            self.assertEqual(plan.representatives[2], 2)  # chain to a fails direct .99 cosine
            self.assertEqual(plan.representatives[3:], [3, 4, 5])
            self.assertFalse(plan.original_bytes_preserved)
            for i, r in enumerate(plan.representatives):
                if i == r:
                    continue
                a, b = rows[i], rows[r]
                cosine = sum(x*y for x, y in zip(a, b)) / math.hypot(*a) / math.hypot(*b)
                ratio = min(math.hypot(*a), math.hypot(*b)) / max(math.hypot(*a), math.hypot(*b))
                self.assertGreaterEqual(cosine, .99)
                self.assertGreaterEqual(ratio, .99)

    def test_layout_only_bytes_exact_signed_zero_and_duplicate_keys(self):
        raw = struct.pack("<6f", 0., 0., -0., 0., 0., 0.)
        keys = {0: "a", 1: "ab", 2: "abc"}
        with StructuralPlan(raw, 2, keys, self.library, mode="layout") as plan:
            self.assertEqual(plan.representatives, [0, 1, 2])
        with StructuralPlan(raw, 2, keys, self.library, mode="exact") as plan:
            self.assertEqual(plan.representatives[1], 1)  # signed zero bytes differ
            self.assertTrue(plan.original_bytes_preserved)
        for bad in ({0: "a", 1: "a"}, {0: "a\0b"}, {0: ""}, {3: "x"}):
            with self.assertRaises(ValueError):
                StructuralPlan(raw, 2, bad, self.library)

    def test_changed_source_and_failed_export_do_not_leave_partial_directory(self):
        from unittest.mock import patch
        raw = struct.pack("<4f", 1., 0., 1., 0.)
        with tempfile.TemporaryDirectory() as tmp, StructuralPlan(raw, 2, {0: "a", 1: "ab"}, self.library) as plan:
            source, output = Path(tmp) / "raw", Path(tmp) / "export"
            source.write_bytes(raw + b"x")
            with self.assertRaisesRegex(ValueError, "differ"):
                plan.export(source, output)
            self.assertFalse(output.exists())
            source.write_bytes(raw)
            with patch("tools.structural_native.pack_native", side_effect=OSError("injected")):
                with self.assertRaisesRegex(OSError, "injected"):
                    plan.export(source, output)
            self.assertFalse(output.exists())
            output.mkdir()
            (output / "keep").write_text("existing")
            with self.assertRaises(FileExistsError):
                plan.export(source, output)
            self.assertTrue((output / "keep").exists())

    def test_compact_export_uses_existing_frozen_embedding_adapter(self):
        try:
            import torch
            from tools.engram_row_adapter import MappedEmbedding
        except ImportError:
            self.skipTest("PyTorch not installed")
        raw = struct.pack("<6f", 0., 0., 1., 0., .996, .089)
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "raw", Path(tmp) / "compact"
            source.write_bytes(raw)
            with StructuralPlan(raw, 2, {1: "a", 2: "ab"}, self.library, mode="joint") as plan:
                plan.export(source, output)
            with MappedEmbedding(output, self.rows_library, native_cache_bytes=1024,
                                 native_cache_policy="rows") as memory:
                ids = torch.tensor([[2, 0, 1, 2]])
                self.assertTrue(torch.equal(memory(ids), torch.tensor([[[1., 0.], [0., 0.], [1., 0.], [1., 0.]]])))
                self.assertTrue(memory.gather.metadata["approximate"])
                self.assertEqual(memory.gather.table.header["n_rows"], 3)

    def test_default_preserves_bytes_and_keeps_unindexed_rows(self):
        raw = struct.pack("<6f", 0., 0., 1., 0., .996, .089)
        with StructuralPlan(raw, 2, {1: "a", 2: "ab"}, self.library) as plan:
            self.assertEqual(plan.mode, "exact")
            self.assertTrue(plan.original_bytes_preserved)
            self.assertEqual(plan.representatives, [0, 1, 2])
        for mode in ("layout", "exact", "joint"):
            with StructuralPlan(raw, 2, {}, self.library, mode=mode) as plan:
                self.assertEqual(plan.representatives, [0, 1, 2])
                self.assertEqual(plan.lookup(["unknown"]), [None])
