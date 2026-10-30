import ctypes
import os
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest
import zlib

from tools.native_row_store import DESC, HEADER, NativeRows, pack_native, resolve_block_rows

ROOT = Path(__file__).resolve().parents[1]


class NativeRowsTest(unittest.TestCase):
    def test_byte_cap_keeps_original_ids_with_compaction_permutation_and_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, packed = Path(tmp) / "raw", Path(tmp) / "table"
            rows = [b"A" * 65, b"B" * 65, b"A" * 65, b"C" * 65, b"D" * 65, b"E" * 65]
            source.write_bytes(b"".join(rows))
            metadata = pack_native(source, packed, 65, 256, False, "xor",
                row_order=[5, 4, 3, 2, 1, 0], representatives=[0, 1, 0, 3, 4, 5], max_block_bytes=130)
            self.assertEqual(metadata['block_rows'], 2)
            self.assertEqual(metadata['blocks'], 3)
            with packed.open('rb') as stream:
                header = HEADER.unpack(stream.read(HEADER.size))
                stream.seek(HEADER.size + header[2] * 8)
                sizes = [DESC.unpack(stream.read(DESC.size))[2] for _ in range(header[-1])]
            self.assertEqual(sizes, [130, 130, 65])
            with NativeRows(packed, self.library) as table:
                self.assertEqual(table.rows([2, 5, 0, 3, 1, 4, 2]), b"".join(rows[i] for i in [2, 5, 0, 3, 1, 4, 2]))
            for cap, expected in ((None, 256), (1, 1), (64, 1), (130, 2), (10**9, 256)):
                self.assertEqual(resolve_block_rows(65, 256, cap), expected)
            for invalid in (0, -1, True, 1.5, '8'):
                with self.assertRaises(ValueError):
                    resolve_block_rows(65, 256, invalid)
            cap_table, explicit_table = Path(tmp) / 'cap', Path(tmp) / 'explicit'
            pack_native(source, cap_table, 65, 256, False, 'xor', max_block_bytes=130)
            pack_native(source, explicit_table, 65, 2, False, 'xor')
            self.assertEqual(cap_table.read_bytes(), explicit_table.read_bytes())

    def test_codec_bytes_count_whole_blocks_chunk_redecodes_cache_hits_and_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, packed = Path(tmp) / 'raw', Path(tmp) / 'table'
            # Repeated fixtures test codecs, not trained-data compression claims.
            source.write_bytes(b'A' * (128 * 8))
            pack_native(source, packed, 128, 4, False, 'xor')
            with packed.open('rb') as stream:
                stream.seek(HEADER.size + 8 * 8)
                descriptors = [DESC.unpack(stream.read(DESC.size)) for _ in range(2)]
            query = (ctypes.c_uint64 * 4)(0, 4, 1, 5)
            out = ctypes.create_string_buffer(4 * 128)
            with NativeRows(packed, self.library, cache_bytes=512, cache_policy='rows') as table:
                table.gather_into(ctypes.addressof(query), 4, ctypes.addressof(out), len(out), max_query_rows=1)
                self.assertEqual(table.last_decoded_bytes, 4 * 512)
                self.assertEqual(table.last_compressed_bytes, 2 * sum(d[1] for d in descriptors))
                table.gather_into(ctypes.addressof(query), 4, ctypes.addressof(out), len(out))
                self.assertEqual(table.last_decoded_bytes, 0)
                self.assertEqual(table.last_compressed_bytes, 0)
                self.assertEqual(table.last_row_cache_hits, 4)
                table.gather_into(0, 0, 0, 0)
                self.assertEqual(table.last_decoded_bytes, 0)
                table.set_cache_bytes(0)
                table.rows([0, 1, 4, 5])
                self.assertEqual(table.last_decoded_bytes, 2 * 512)
                self.assertEqual(table.last_compressed_bytes, sum(d[1] for d in descriptors))
            source.write_bytes(random.Random(42).randbytes(128 * 8))
            metadata = pack_native(source, packed, 128, 4, False)
            self.assertEqual(metadata['raw_blocks'], 2)
            with NativeRows(packed, self.library) as table:
                table.rows([0, 4])
                self.assertEqual(table.last_decoded_bytes, 0)
                self.assertEqual(table.last_compressed_bytes, 0)

    def test_bag_calls_aggregate_codec_bytes_and_reset_on_rejected_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, packed = Path(tmp) / 'raw', Path(tmp) / 'table'
            source.write_bytes(struct.pack('<32f', *([1.0] * 32)))
            pack_native(source, packed, 16, 4, False)
            with NativeRows(packed, self.library) as table:
                ids = (ctypes.c_uint64 * 4)(0, 4, 1, 5)
                offsets = (ctypes.c_uint64 * 3)(0, 2, 4)
                out = (ctypes.c_float * 8)()
                table.embedding_bag_into(ctypes.addressof(ids), 4, ctypes.addressof(offsets), 2,
                    1, False, ctypes.c_uint64(-1).value, 1, ctypes.addressof(out), 8)
                self.assertEqual(table.last_decoded_bytes, 4 * 64)
                ids[3] = 8
                gathered = ctypes.create_string_buffer(64)
                with self.assertRaises(ValueError):
                    table.gather_into(ctypes.addressof(ids), 4, ctypes.addressof(gathered), len(gathered))
                self.assertEqual(table.last_decoded_bytes, 0)

    def test_chunked_query_preserves_bytes_order_and_aggregates_all_counters(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, packed = Path(tmp) / "raw", Path(tmp) / "table"
            rows = [bytes([i]) * 128 for i in range(16)]
            source.write_bytes(b"".join(rows))
            pack_native(source, packed, 128, 4, False)
            ids = [0, 4, 0, 4, 1, 5]
            query = (ctypes.c_uint64 * len(ids))(*ids)
            output = ctypes.create_string_buffer(len(ids) * 128)
            with NativeRows(packed, self.library) as table:
                for limit, reads in ((None, 2), (1, 6), (3, 4), (100, 2)):
                    table.gather_into(ctypes.addressof(query), len(ids), ctypes.addressof(output), len(output),
                                      max_query_rows=limit)
                    self.assertEqual(output.raw, b"".join(rows[i] for i in ids))
                    self.assertEqual(table.last_block_reads, reads)
                    self.assertEqual(table.last_decompressions, reads)
                table.set_cache_bytes(1024)
                table.gather_into(ctypes.addressof(query), len(ids), ctypes.addressof(output), len(output), max_query_rows=3)
                self.assertEqual(table.last_block_reads, 2)
                self.assertEqual(table.last_decompressions, 2)
                self.assertEqual(table.last_cache_hits, 2)
                table.gather_into(0, 0, 0, 0, max_query_rows=3)
                self.assertEqual(table.last_block_reads, 0)
                self.assertEqual(table.last_cache_hits, 0)

    def test_chunked_query_rejects_limits_capacities_and_late_invalid_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, packed = Path(tmp) / "raw", Path(tmp) / "table"
            source.write_bytes(b"A" * 64)
            pack_native(source, packed, 8, 2, False)
            query = (ctypes.c_uint64 * 3)(0, 1, 8)
            output = ctypes.create_string_buffer(24)
            with NativeRows(packed, self.library) as table:
                for invalid in (0, -1, True, 1.5, ctypes.c_size_t(-1).value + 1):
                    with self.assertRaises(ValueError):
                        table.gather_into(ctypes.addressof(query), 3, ctypes.addressof(output), 24,
                                          max_query_rows=invalid)
                with self.assertRaisesRegex(ValueError, "capacity"):
                    table.gather_into(ctypes.addressof(query), 3, ctypes.addressof(output), 23, max_query_rows=2)
                with self.assertRaisesRegex(ValueError, "row ID"):
                    table.gather_into(ctypes.addressof(query), 3, ctypes.addressof(output), 24, max_query_rows=2)
                # Error output can be partial and must be discarded.
                self.assertEqual(table.last_block_reads, 1)
                function = table.lib.meft_rows_gather_chunked
                error = ctypes.create_string_buffer(256)
                self.assertNotEqual(function(table.handle, 0, 0, 0, 0, 0, error, len(error)), 0)
                self.assertNotEqual(function(table.handle, 0, 3, ctypes.addressof(output), 24, 2, error, len(error)), 0)
                self.assertNotEqual(function(0, 0, 0, 0, 0, 2, error, len(error)), 0)

    def test_small_queries_and_disabled_chunking_keep_older_library_compatibility(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            source, packed = Path(tmp) / "raw", Path(tmp) / "table"
            source.write_bytes(b"A" * 64)
            pack_native(source, packed, 8, 2, False)
            with NativeRows(packed, self.library) as table:
                real_library = table.lib
                class OlderLibrary:
                    def __getattr__(self, name):
                        if name == "meft_rows_gather_chunked":
                            raise AttributeError(name)
                        return getattr(real_library, name)
                query = (ctypes.c_uint64 * 2)(0, 1)
                output = ctypes.create_string_buffer(16)
                with patch.object(table, "lib", OlderLibrary()):
                    for limit in (None, 2):
                        table.gather_into(ctypes.addressof(query), 2, ctypes.addressof(output), 16, max_query_rows=limit)
                        self.assertEqual(output.raw, b"A" * 16)
                    with self.assertRaisesRegex(ValueError, "rebuilding"):
                        table.gather_into(ctypes.addressof(query), 2, ctypes.addressof(output), 16, max_query_rows=1)

    def test_atomic_replacement_keeps_old_mmap_readers_on_original_bytes(self):
        rng = random.Random(27)
        with tempfile.TemporaryDirectory() as tmp:
            source, packed = Path(tmp) / "raw", Path(tmp) / "table.men"
            old = rng.randbytes(64 * 8)
            new = rng.randbytes(64 * 8)
            source.write_bytes(old)
            pack_native(source, packed, 64, 4, False)
            packed.chmod(0o640)
            with NativeRows(packed, self.library) as previous:
                original_inode = packed.stat().st_ino
                source.write_bytes(new)
                pack_native(source, packed, 64, 4, False)
                self.assertNotEqual(original_inode, packed.stat().st_ino)
                self.assertEqual(packed.stat().st_mode & 0o777, 0o640)
                # Uncached C reads happen after publication, not just before it.
                self.assertEqual(previous.rows(range(8)), old)
                with NativeRows(packed, self.library) as current:
                    self.assertEqual(current.rows(range(8)), new)
            self.assertFalse(list(Path(tmp).glob(".table.men.*.tmp")))

    def test_failed_flush_or_replace_preserves_target_and_cleans_temporary_file(self):
        from unittest.mock import patch
        for failure in ("fsync", "replace"):
            for existing in (False, True):
                with self.subTest(failure=failure, existing=existing), tempfile.TemporaryDirectory() as tmp:
                    source, packed = Path(tmp) / "raw", Path(tmp) / "table.men"
                    source.write_bytes(b"A" * 64)
                    if existing:
                        pack_native(source, packed, 8, 4, False)
                    original = packed.read_bytes() if existing else None
                    source.write_bytes(b"B" * 64)
                    with patch("tools.native_row_store.os." + failure, side_effect=OSError("injected export failure")):
                        with self.assertRaisesRegex(OSError, "injected"):
                            pack_native(source, packed, 8, 4, False)
                    self.assertEqual(packed.read_bytes() if packed.exists() else None, original)
                    self.assertFalse(list(Path(tmp).glob(".table.men.*.tmp")))

    def test_failed_partial_write_never_publishes_a_partial_table(self):
        from unittest.mock import patch
        real_temporary = tempfile.NamedTemporaryFile

        def broken_writer(*args, **kwargs):
            result = real_temporary(*args, **kwargs)
            write = result.write
            calls = 0

            def fail_after_header(data):
                nonlocal calls
                calls += 1
                if calls > 1:
                    raise OSError("partial write")
                return write(data)

            result.write = fail_after_header
            return result

        with tempfile.TemporaryDirectory() as tmp:
            source, packed = Path(tmp) / "raw", Path(tmp) / "table.men"
            source.write_bytes(b"old rows" * 8)
            pack_native(source, packed, 8, 4, False)
            original = packed.read_bytes()
            source.write_bytes(b"new rows" * 8)
            with patch("tools.native_row_store.tempfile.NamedTemporaryFile", side_effect=broken_writer):
                with self.assertRaisesRegex(OSError, "partial write"):
                    pack_native(source, packed, 8, 4, False)
            self.assertEqual(packed.read_bytes(), original)
            self.assertFalse(list(Path(tmp).glob(".table.men.*.tmp")))

    def test_export_to_source_hardlink_does_not_modify_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, packed = Path(tmp) / "raw", Path(tmp) / "table.men"
            raw = b"hardlink" * 8
            source.write_bytes(raw)
            os.link(source, packed)
            pack_native(source, packed, 8, 4, False)
            self.assertEqual(source.read_bytes(), raw)
            with NativeRows(packed, self.library) as table:
                self.assertEqual(table.rows(range(8)), raw)

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.library = Path(cls.tmp.name) / "librows.so"
        subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-O2", "-Wall", "-Wextra",
                        "-Werror", "-fPIC", "-shared", str(ROOT / "native/meft_rows.c"),
                        "-lz", "-o", str(cls.library)], check=True)
        cls.example = Path(cls.tmp.name) / "gather"
        subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-Wall", "-Wextra", "-Werror",
                        "-I" + str(ROOT / "native"), str(ROOT / "examples/native_gather.c"),
                        str(ROOT / "native/meft_rows.c"), "-lz", "-o", str(cls.example)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_native_preserves_arbitrary_bytes_ids_and_batches(self):
        rng = random.Random(3)
        for width in (1, 3, 64, 256):
            for dedup in (False, True):
                with self.subTest(width=width, dedup=dedup), tempfile.TemporaryDirectory() as tmp:
                    # Zero blocks compress; random blocks select raw fallback.
                    rows = [b"\0" * width for _ in range(6)]
                    rows += [rng.getrandbits(width * 8).to_bytes(width, "little") for _ in range(31)]
                    rows += [rows[1], rows[12], rows[12]]
                    source, packed = Path(tmp) / "raw", Path(tmp) / "table.men"
                    source.write_bytes(b"".join(rows))
                    pack_native(source, packed, width, 4, dedup)
                    with NativeRows(packed, self.library) as table:
                        ids = [rng.randrange(len(rows)) for _ in range(200)]
                        self.assertEqual(table.rows(ids), b"".join(rows[i] for i in ids))
                        self.assertEqual(table.rows(range(len(rows))), b"".join(rows))
                        self.assertEqual(table.rows([]), b"")
                        with self.assertRaises(IndexError):
                            table.row(-1)
                        bad_id = (ctypes.c_uint64 * 1)(len(rows))
                        out = ctypes.create_string_buffer(width)
                        with self.assertRaisesRegex(ValueError, "row ID"):
                            table.gather_into(ctypes.addressof(bad_id), 1, ctypes.addressof(out), width)
                        valid_id = (ctypes.c_uint64 * 1)(0)
                        with self.assertRaisesRegex(ValueError, "capacity"):
                            table.gather_into(ctypes.addressof(valid_id), 1, ctypes.addressof(out), width - 1)
                    result = subprocess.run([str(self.example), str(packed), "0", "12", "0"],
                                            check=True, capture_output=True, text=True)
                    self.assertIn("id=12 bytes=" + rows[12].hex(), result.stdout)

    def test_native_coalesces_blocks_and_rejects_bad_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, packed = Path(tmp) / "raw", Path(tmp) / "table.men"
            rows = [bytes([i]) * 128 for i in range(16)]
            source.write_bytes(b"".join(rows))
            pack_native(source, packed, 128, 4, False)
            with NativeRows(packed, self.library) as table:
                table.rows([0, 4, 0, 4, 1, 5])
                self.assertEqual(table.last_block_reads, 2)
                self.assertEqual(table.last_decompressions, 2)
            data = packed.read_bytes()
            for mutated in (data[:20], data[:-1], b"BADMAGIC" + data[8:], data + b"extra"):
                packed.write_bytes(mutated)
                with self.assertRaises(ValueError):
                    NativeRows(packed, self.library)
            # Corrupt the first queried forwarding ID: reject on gather.
            changed = bytearray(data)
            struct.pack_into("<Q", changed, 56, 99999)
            packed.write_bytes(changed)
            with NativeRows(packed, self.library) as table, self.assertRaisesRegex(ValueError, "forwarding"):
                table.row(0)
            # Corrupt a compressed payload while keeping structural offsets.
            changed = bytearray(data)
            offset = struct.unpack_from("<Q", data, 56 + len(rows) * 8)[0]
            changed[offset] ^= 255
            packed.write_bytes(changed)
            with NativeRows(packed, self.library) as table, self.assertRaisesRegex(ValueError, "decompression"):
                table.row(0)

    def test_cache_reuses_exact_blocks_and_respects_lru_byte_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, packed = Path(tmp) / "raw", Path(tmp) / "table.men"
            rows = [bytes([i]) * 128 for i in range(19)]
            source.write_bytes(b"".join(rows))
            pack_native(source, packed, 128, 4, False, "xor")
            with NativeRows(packed, self.library, cache_bytes=1024) as table:
                for ids, hits, decodes in (([0, 4, 0], 0, 2), ([0], 1, 0),
                                          ([8], 0, 1), ([0], 1, 0), ([4], 0, 1)):
                    self.assertEqual(table.rows(ids), b"".join(rows[i] for i in ids))
                    self.assertEqual(table.last_cache_hits, hits)
                    self.assertEqual(table.last_decompressions, decodes)
                    self.assertLessEqual(table.cache_bytes, 1024)
                # Tail block occupies actual decoded size, not max block size.
                table.set_cache_bytes(384)
                table.rows([16]); self.assertEqual(table.cache_bytes, 384)
                table.rows([16]); self.assertEqual(table.last_cache_hits, 1)
                table.rows([0]); self.assertEqual(table.cache_bytes, 384)
                self.assertEqual(table.last_decompressions, 1)  # Oversized block bypasses cache.
                table.set_cache_bytes(0)
                self.assertEqual(table.cache_bytes, 0)
                table.rows([16]); table.rows([16])
                self.assertEqual(table.last_decompressions, 1)
                for invalid in (-1, True, 1.0, ctypes.c_size_t(-1).value + 1):
                    with self.assertRaises(ValueError):
                        table.set_cache_bytes(invalid)
            with self.assertRaisesRegex(ValueError, "closed"):
                table.set_cache_bytes(0)

    def test_cache_raw_bypass_and_failed_decode_are_not_cached(self):
        rng = random.Random(53)
        with tempfile.TemporaryDirectory() as tmp:
            source, packed = Path(tmp) / "raw", Path(tmp) / "table.men"
            source.write_bytes(rng.randbytes(256))
            pack_native(source, packed, 64, 4, False)
            with NativeRows(packed, self.library, 1024) as table:
                table.rows([0]); table.rows([0])
                self.assertEqual(table.cache_bytes, 0)
                self.assertEqual(table.last_decompressions, 0)
                self.assertEqual(table.last_cache_hits, 0)
            source.write_bytes(b"\0" * 256)
            pack_native(source, packed, 64, 4, False)
            data = bytearray(packed.read_bytes())
            offset = struct.unpack_from("<Q", data, 56 + 4 * 8)[0]
            data[offset] ^= 255
            packed.write_bytes(data)
            with NativeRows(packed, self.library, 1024) as table:
                for _ in range(2):
                    with self.assertRaisesRegex(ValueError, "decompression"):
                        table.rows([0])
                    self.assertEqual(table.cache_bytes, 0)
                    self.assertEqual(table.last_cache_hits, 0)

    def test_cache_uses_budget_beyond_64_entries_and_invalid_query_is_atomic(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, packed = Path(tmp) / "raw", Path(tmp) / "table.men"
            rows = [bytes([i % 256]) * 128 for i in range(300)]
            source.write_bytes(b"".join(rows))
            pack_native(source, packed, 128, 1, False)
            with NativeRows(packed, self.library, 1024 * 1024) as table:
                self.assertEqual(table.cache_bytes, 0)  # Budget allocation is lazy.
                self.assertEqual(table.rows(range(300)), b"".join(rows))
                self.assertEqual(table.cache_bytes, 300 * 128)
                self.assertEqual(table.rows([0, 299, 0, 128, 299]), b"".join(rows[i] for i in [0, 299, 0, 128, 299]))
                self.assertEqual(table.last_cache_hits, 3)  # Count blocks, not repeated rows.
                self.assertEqual(table.last_decompressions, 0)
                ids = (ctypes.c_uint64 * 2)(299, 300)
                out = ctypes.create_string_buffer(256)
                original_output = out.raw
                with self.assertRaisesRegex(ValueError, "row ID"):
                    table.gather_into(ctypes.addressof(ids), 2, ctypes.addressof(out), 256)
                self.assertEqual(out.raw, original_output)
                self.assertEqual(table.rows([299]), rows[299])
                self.assertEqual(table.last_cache_hits, 1)
                table.rows([0]); self.assertEqual(table.last_decompressions, 0)

    def test_cache_consumes_later_hits_before_earlier_misses_evict_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, packed = Path(tmp) / "raw", Path(tmp) / "table.men"
            rows = [bytes([i]) * 128 for i in range(10)]
            source.write_bytes(b"".join(rows))
            pack_native(source, packed, 128, 1, False)
            with NativeRows(packed, self.library, cache_bytes=256) as table:
                table.rows([8, 9])
                self.assertEqual(table.rows([0, 8, 1, 9, 8]), b"".join(rows[i] for i in [0, 8, 1, 9, 8]))
                self.assertEqual(table.last_cache_hits, 2)
                self.assertEqual(table.last_decompressions, 2)
                self.assertEqual(table.cache_bytes, 256)
                self.assertEqual(table.rows([0, 1, 0]), rows[0] + rows[1] + rows[0])
                self.assertEqual(table.last_decompressions, 0)

    def test_cache_hash_growth_eviction_and_budget_reset_match_lru_model(self):
        from collections import OrderedDict
        rng = random.Random(76)
        with tempfile.TemporaryDirectory() as tmp:
            source, packed = Path(tmp) / "raw", Path(tmp) / "table.men"
            rows = [i.to_bytes(2, "little") * 64 for i in range(300)]
            source.write_bytes(b"".join(rows))
            pack_native(source, packed, 128, 1, False)
            with NativeRows(packed, self.library) as table:
                for capacity in (150, 65, 1, 0, 300):
                    table.set_cache_bytes(capacity * 128)
                    self.assertEqual(table.cache_bytes, 0)
                    model = OrderedDict()
                    # Growth, collisions, eviction and reset use a deterministic replay.
                    for row in [*range(300), *(rng.randrange(300) for _ in range(400))]:
                        hit = row in model
                        self.assertEqual(table.rows([row, row]), rows[row] * 2)
                        self.assertEqual(table.last_cache_hits, int(hit))
                        self.assertEqual(table.last_decompressions, int(not hit))
                        if capacity:
                            model[row] = None
                            model.move_to_end(row)
                            if len(model) > capacity:
                                model.popitem(last=False)
                        self.assertEqual(table.cache_bytes, len(model) * 128)

    def test_row_cache_retains_sparse_queries_and_forwarded_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, packed = Path(tmp) / "raw", Path(tmp) / "table.men"
            # Opaque bit patterns, odd width and original forwarding aliases.
            rows = [i.to_bytes(2, "little") * 31 + bytes([i]) for i in range(80)]
            source.write_bytes(b"".join(rows + rows))
            pack_native(source, packed, 63, 8, True, "xor")
            with NativeRows(packed, self.library, cache_bytes=4 * 63, cache_policy="rows") as table:
                ids = [0, 64, 80, 32, 0]
                self.assertEqual(table.rows(ids), b"".join(rows[i % 80] for i in ids))
                self.assertEqual(table.last_decompressions, 3)
                self.assertEqual(table.cache_bytes, 3 * 63)  # Original alias 80 shares row 0.
                self.assertEqual(table.last_row_cache_hits, 0)
                self.assertEqual(table.rows(ids[::-1]), b"".join(rows[i % 80] for i in ids[::-1]))
                self.assertEqual(table.last_decompressions, 0)
                self.assertEqual(table.last_row_cache_hits, 3)
                self.assertEqual(table.last_cache_hits, 0)
                table.rows([8, 16])  # Evicts two original rows under the exact budget.
                self.assertEqual(table.cache_bytes, 4 * 63)
                table.set_cache_policy("blocks")
                self.assertEqual(table.cache_bytes, 0)
                table.rows([0]); table.rows([0])
                self.assertEqual(table.last_decompressions, 1)  # Whole block exceeds budget.
                for bad in ("unknown", 1, None):
                    with self.assertRaisesRegex(ValueError, "policy"):
                        table.set_cache_policy(bad)
                self.assertEqual(table.lib.meft_rows_set_cache_policy(table.handle, 2), -1)
                table.set_cache_policy("rows")
                table.rows([0]); table.rows([80])
                self.assertEqual(table.last_row_cache_hits, 1)
                table.set_cache_bytes(62)
                table.rows([0]); table.rows([0])
                self.assertEqual(table.cache_bytes, 0)
                self.assertEqual(table.last_decompressions, 1)

    def test_row_cache_chunk_counters_and_failed_queries_preserve_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, packed = Path(tmp) / "raw", Path(tmp) / "table.men"
            rows = [bytes([i]) * 128 for i in range(20)]
            source.write_bytes(b"".join(rows))
            pack_native(source, packed, 128, 4, False)
            with NativeRows(packed, self.library, cache_bytes=256, cache_policy="rows") as table:
                table.rows([0, 1])
                ids = (ctypes.c_uint64 * 4)(0, 1, 0, 1)
                out = ctypes.create_string_buffer(512)
                table.gather_into(ctypes.addressof(ids), 4, ctypes.addressof(out), 512, max_query_rows=1)
                self.assertEqual(out.raw, (rows[0] + rows[1]) * 2)
                self.assertEqual(table.last_row_cache_hits, 4)
                self.assertEqual(table.last_decompressions, 0)
                bad = (ctypes.c_uint64 * 2)(0, 20)
                before = out.raw
                with self.assertRaisesRegex(ValueError, "row ID"):
                    table.gather_into(ctypes.addressof(bad), 2, ctypes.addressof(out), 512)
                self.assertEqual(out.raw, before)
                self.assertEqual(table.cache_bytes, 256)
                table.gather_into(0, 0, 0, 0)
                self.assertEqual(table.last_row_cache_hits, 0)

    def test_row_cache_failed_decode_never_publishes_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, packed = Path(tmp) / "raw", Path(tmp) / "table.men"
            source.write_bytes(b"\0" * 512)
            pack_native(source, packed, 128, 4, False)
            data = bytearray(packed.read_bytes())
            offset = struct.unpack_from("<Q", data, 56 + 4 * 8)[0]
            data[offset] ^= 255
            packed.write_bytes(data)
            with NativeRows(packed, self.library, cache_bytes=128, cache_policy="rows") as table:
                for _ in range(2):
                    with self.assertRaisesRegex(ValueError, "decompression"):
                        table.rows([0])
                    self.assertEqual(table.cache_bytes, 0)
                    self.assertEqual(table.last_row_cache_hits, 0)

    def test_cache_preserves_older_library_loading_when_disabled(self):
        from unittest.mock import patch
        class LegacyLibrary:
            def __init__(self, library):
                self.library = library
            def __getattr__(self, name):
                if name in ("meft_rows_set_cache_bytes", "meft_rows_set_cache_policy", "meft_rows_last_row_cache_hits"):
                    raise AttributeError(name)
                return getattr(self.library, name)
        with tempfile.TemporaryDirectory() as tmp:
            source, packed = Path(tmp) / "raw", Path(tmp) / "table.men"
            source.write_bytes(b"abcabc")
            pack_native(source, packed, 3, 2)
            legacy = LegacyLibrary(ctypes.CDLL(str(self.library)))
            with patch("tools.native_row_store.ctypes.CDLL", return_value=legacy):
                with NativeRows(packed, self.library) as store:
                    self.assertEqual(store.rows([0, 1]), b"abcabc")
                    self.assertEqual(store.last_row_cache_hits, 0)
                with self.assertRaisesRegex(ValueError, "rebuilding"):
                    NativeRows(packed, self.library, cache_bytes=64)
                with self.assertRaisesRegex(ValueError, "rebuilding"):
                    NativeRows(packed, self.library, cache_policy="rows")

    def test_xor_codec_adapts_and_preserves_reordered_original_ids(self):
        rng = random.Random(23)
        width = 256
        base = rng.randbytes(width)
        # Individually high-entropy rows share most bits; XOR residuals compress.
        related = [base[:i] + bytes([base[i] ^ 1]) + base[i + 1:] for i in range(8)]
        rows = related + [b"\0" * width] * 8 + [rng.randbytes(width) for _ in range(8)]
        for dedup in (False, True):
            with self.subTest(dedup=dedup), tempfile.TemporaryDirectory() as tmp:
                source, packed, plain = (Path(tmp) / name for name in ("raw", "xor.men", "plain.men"))
                source.write_bytes(b"".join(rows))
                # Reverse within groups to exercise original-ID forwarding.
                permutation = [i for start in range(0, len(rows), 8)
                               for i in reversed(range(start, start + 8))]
                result = pack_native(source, packed, width, 8, dedup, "xor", permutation)
                control = pack_native(source, plain, width, 8, dedup, row_order=permutation)
                self.assertLessEqual(result["artifact_bytes"], control["artifact_bytes"])
                self.assertGreater(result["xor_blocks"], 0)
                self.assertGreater(result["raw_blocks"], 0)
                if not dedup:
                    self.assertGreater(result["zlib_blocks"], 0)
                with NativeRows(packed, self.library) as table:
                    self.assertEqual(table.rows(range(len(rows))), b"".join(rows))
                    ids = [23, 0, 7, 8, 0, 16, 1, 23]
                    self.assertEqual(table.rows(ids), b"".join(rows[i] for i in ids))
                    self.assertEqual(table.last_block_reads, 3)
                data = packed.read_bytes()
                # Version 1 must reject a version 2 codec rather than misdecode it.
                changed = bytearray(data)
                struct.pack_into("<Q", changed, 8, 1)
                packed.write_bytes(changed)
                with self.assertRaisesRegex(ValueError, "descriptor"):
                    NativeRows(packed, self.library)
                changed = bytearray(data)
                struct.pack_into("<Q", changed, 8, 3)
                packed.write_bytes(changed)
                with self.assertRaisesRegex(ValueError, "header"):
                    NativeRows(packed, self.library)
                # Corrupt the XOR zlib stream, keeping descriptor offsets valid.
                changed = bytearray(data)
                offset = struct.unpack_from("<Q", data, 56 + len(rows) * 8)[0]
                changed[offset] ^= 255
                packed.write_bytes(changed)
                with NativeRows(packed, self.library) as table:
                    with self.assertRaisesRegex(ValueError, "decompression"):
                        table.row(0)

    def test_xor_empty_partial_blocks_and_invalid_permutations(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, packed = Path(tmp) / "raw", Path(tmp) / "table.men"
            for rows in ([], [b"abc"], [b"abc", b"abd", b"abe"]):
                source.write_bytes(b"".join(rows))
                pack_native(source, packed, 3, 2, False, "xor", list(reversed(range(len(rows)))))
                with NativeRows(packed, self.library) as table:
                    self.assertEqual(table.rows(range(len(rows))), b"".join(rows))
            for permutation in ([0, 1], [0, 0, 2], [0, 1, 3], [0, 1, -1], [0, 1, True]):
                with self.assertRaisesRegex(ValueError, "permutation"):
                    pack_native(source, packed, 3, row_order=permutation)
            with self.assertRaisesRegex(ValueError, "predictor"):
                pack_native(source, packed, 3, predictor="invalid")

    def test_xor_decoder_handles_unaligned_rows_and_word_tails(self):
        rng = random.Random(37)
        for width in (127, 255, 257):
            with self.subTest(width=width), tempfile.TemporaryDirectory() as tmp:
                base = rng.randbytes(width)
                rows = [base[:i] + bytes([base[i] ^ 1]) + base[i + 1:] for i in range(17)]
                source, packed = Path(tmp) / "raw", Path(tmp) / "table.men"
                source.write_bytes(b"".join(rows))
                result = pack_native(source, packed, width, 8, False, "xor")
                self.assertGreater(result["xor_blocks"], 0)
                with NativeRows(packed, self.library) as table:
                    self.assertEqual(table.rows(range(17)), b"".join(rows))

    def test_integer_xor_matches_reference_bytes_with_leading_and_trailing_zeros(self):
        rng = random.Random(43)
        for width in (1, 3, 127, 512):
            with self.subTest(width=width), tempfile.TemporaryDirectory() as tmp:
                base = b"\0" + rng.randbytes(width - 1)
                rows = [base] * 8 + [b"\0" * width]
                source, packed = Path(tmp) / "raw", Path(tmp) / "table.men"
                source.write_bytes(b"".join(rows))
                pack_native(source, packed, width, 8, False, "xor")
                data = packed.read_bytes()
                for block in range(2):
                    offset, stored, raw_size, codec = struct.unpack_from("<4Q", data, 56 + 8 * len(rows) + block * 32)
                    raw = b"".join(rows[block * 8:(block + 1) * 8])
                    residual = bytearray(raw)
                    for i in range(width, len(raw)):
                        residual[i] ^= raw[i - width]
                    candidates = [raw, zlib.compress(raw), zlib.compress(residual)]
                    # A one-row block does not need a predictor; ordinary ties win.
                    expected_codec = min(range(3), key=lambda i: len(candidates[i]))
                    self.assertEqual(codec, expected_codec)
                    self.assertEqual(data[offset:offset + stored], candidates[expected_codec])
                    self.assertEqual(raw_size, len(raw))

    def test_xor_torch_reconstructs_nonfinite_bit_patterns(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional PyTorch unavailable")
        from tools.engram_row_adapter import export_multihead, MappedMultiHeadEmbedding
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                torch.manual_seed(19)
                # Include NaN payloads, signed zero and infinities as opaque bits.
                bits = torch.randint(0, 256, (1, 512), dtype=torch.uint8).repeat(8, 1)
                special = {torch.float32: "000000800000807f0100c07f",
                           torch.float16: "0080007c017e", torch.bfloat16: "0080807f817f"}[dtype]
                prefix = bytes.fromhex(special)
                bits[:, :len(prefix)] = torch.tensor(list(prefix), dtype=torch.uint8)
                for i in range(8):
                    bits[i, 64 + i] ^= 1
                weight = bits.view(dtype)
                meta = export_multihead(weight, [3, 5], tmp, 8, backend="native", predictor="xor")
                table = Path(tmp) / "table.men"
                descriptor = 56 + 8 * len(weight)
                self.assertEqual(struct.unpack_from("<Q", table.read_bytes(), descriptor + 24)[0], 2)
                with MappedMultiHeadEmbedding(tmp, native_library=self.library, native_cache_bytes=4096) as mapped:
                    ids = torch.tensor([[2, 4], [0, 0], [2, 4]])
                    expected = weight[ids + torch.tensor([0, 3])]
                    self.assertTrue(torch.equal(mapped(ids).view(torch.uint8), expected.view(torch.uint8)))
                    self.assertEqual(mapped.table.last_decompressions, 1)
                    self.assertTrue(torch.equal(mapped(ids).view(torch.uint8), expected.view(torch.uint8)))
                    self.assertEqual(mapped.table.last_decompressions, 0)
                    self.assertEqual(mapped.table.last_cache_hits, 1)
                self.assertEqual(meta["predictor"], "xor")

    def test_native_torch_adapter_direct_output(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional PyTorch unavailable")
        from tools.engram_row_adapter import export_multihead, MappedMultiHeadEmbedding
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                torch.manual_seed(7)
                weight = torch.randn(10, 9, dtype=dtype)
                weight[8] = weight[0]
                weight[1, 0] = -0.0
                export_multihead(weight, [3, 5, 2], tmp, 2, backend="native", predictor="xor")
                ids = torch.tensor([[[2, 4, 1], [0, 0, 0]], [[1, 2, 0], [2, 4, 1]]])
                with MappedMultiHeadEmbedding(tmp, native_library=self.library) as mapped:
                    for query in (ids, ids.transpose(0, 1), ids.to(torch.int32), ids[:0]):
                        original = torch.nn.functional.embedding(query.to(torch.int64) + torch.tensor([0, 3, 8]), weight)
                        actual = mapped(query)
                        self.assertEqual(actual.shape, original.shape)
                        self.assertTrue(torch.equal(actual.view(torch.uint8), original.contiguous().view(torch.uint8)))

    def test_ordinary_embedding_and_max_norm_boundary(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional PyTorch unavailable")
        from tools.engram_row_adapter import MappedEmbedding, export_embedding
        with tempfile.TemporaryDirectory() as tmp:
            original = torch.nn.Embedding(17, 7, padding_idx=0).eval()
            export_embedding(original, tmp, block_rows=4)
            with MappedEmbedding(tmp, self.library) as mapped:
                for ids in (torch.tensor([[0, 16, 0], [5, 2, 7]]), torch.tensor(2),
                            torch.empty((0, 3), dtype=torch.int64)):
                    self.assertTrue(torch.equal(mapped(ids).view(torch.uint8), original(ids).view(torch.uint8)))
                with self.assertRaises(RuntimeError):
                    mapped.train()
            with self.assertRaises(ValueError):
                export_embedding(torch.nn.Embedding(17, 7, max_norm=1), Path(tmp) / "maxnorm")

    def test_small_causal_transformer_logits_match_with_untied_embedding(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional PyTorch unavailable")
        from tools.engram_row_adapter import MappedEmbedding, export_embedding

        class TinyDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(31, 16)
                layer = torch.nn.TransformerEncoderLayer(16, 2, 32, dropout=0, batch_first=True)
                self.blocks = torch.nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
                self.lm_head = torch.nn.Linear(16, 31, bias=False)  # deliberately untied

            def forward(self, ids):
                mask = torch.triu(torch.ones(ids.shape[-1], ids.shape[-1], dtype=torch.bool), diagonal=1)
                return self.lm_head(self.blocks(self.embedding(ids), mask=mask))

        with tempfile.TemporaryDirectory() as tmp, torch.inference_mode():
            torch.manual_seed(7)
            model = TinyDecoder().eval()
            ids = torch.tensor([[0, 1, 2, 3, 4], [4, 4, 2, 0, 30]])
            before = model(ids)
            export_embedding(model.embedding, tmp, 4)
            with MappedEmbedding(tmp, self.library) as replacement:
                model.embedding = replacement
                after = model(ids)
                self.assertTrue(torch.equal(before.view(torch.uint8), after.view(torch.uint8)))


if __name__ == "__main__":
    unittest.main()
