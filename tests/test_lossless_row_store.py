import random
import struct
import tempfile
import unittest
from pathlib import Path

from tools.lossless_row_store import MappedRows, pack_rows


class LosslessRowsTest(unittest.TestCase):
    def test_every_original_id_preserves_bytes_and_query_order(self):
        # Include signed zero, different NaN payloads, and equal-norm/opposite
        # float vectors. Numerical similarity must never decide byte equality.
        rows = [struct.pack("<II", 0, 0x80000000),
                struct.pack("<II", 0x7FC00001, 0x7FC00002),
                struct.pack("<ff", 1, 0), struct.pack("<ff", -1, 0)]
        rng = random.Random(7)
        rows.extend(rng.getrandbits(64).to_bytes(8, "little") for _ in range(11))
        rows += [rows[0], rows[2], rows[-1], rows[1]]
        with tempfile.TemporaryDirectory() as tmp:
            source, packed = Path(tmp) / "raw", Path(tmp) / "packed"
            source.write_bytes(b"".join(rows))
            report = pack_rows(source, packed, 8, block_rows=3)
            self.assertEqual(report["unique_rows"], 15)
            for cache_blocks in (0, 1, 8):
                with MappedRows(packed, cache_blocks) as table:
                    for row_id, original in enumerate(rows):
                        self.assertEqual(table.row(row_id), original)
                    query = [18, 0, 4, 18, 2, 1, 0]
                    self.assertEqual(table.rows(query), b"".join(rows[i] for i in query))
                    self.assertLessEqual(len(table.cache), cache_blocks)
                    for bad in (-1, len(rows), 1.5):
                        with self.assertRaises(IndexError):
                            table.row(bad)

    def test_rejects_partial_rows_and_source_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "raw"
            source.write_bytes(b"abc")
            with self.assertRaises(ValueError):
                pack_rows(source, Path(tmp) / "out", 2)
            with self.assertRaises(ValueError):
                pack_rows(source, source, 1)

    def test_truncated_payload_is_not_returned_as_a_vector(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, packed = Path(tmp) / "raw", Path(tmp) / "packed"
            source.write_bytes(b"abcdefgh")
            pack_rows(source, packed, 8)
            packed.write_bytes(packed.read_bytes()[:-2])
            with MappedRows(packed) as table, self.assertRaises(ValueError):
                table.row(0)


if __name__ == "__main__":
    unittest.main()
