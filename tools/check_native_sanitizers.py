"""Compile ASan/UBSan driver and check valid plus deterministically mutated files."""
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.native_row_store import pack_native


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        driver = tmp / "check"
        subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-g", "-O1",
                        "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                        str(ROOT / "native/check_native.c"), str(ROOT / "native/meft_rows.c"),
                        "-lz", "-o", str(driver)], check=True)
        source, table = tmp / "raw", tmp / "table.men"
        rng = random.Random(7)
        base = rng.randbytes(256)
        source.write_bytes(b"".join(base[:i] + bytes([base[i] ^ 1]) + base[i + 1:]
                                  for i in range(128)))
        for predictor in ("none", "xor"):
            packed = pack_native(source, table, 256, 8, False, predictor)
            if predictor == "xor" and not packed["xor_blocks"]:
                raise AssertionError("sanitizer fixture must exercise native XOR decoding")
            valid = table.read_bytes()
            subprocess.run([str(driver), str(table)], check=True)
            for i in range(64):
                raw = bytearray(valid)
                for _ in range(1 + i % 3):
                    raw[rng.randrange(len(raw))] ^= rng.randrange(1, 256)
                if i % 4 == 0:
                    raw = raw[:rng.randrange(len(raw))]
                table.write_bytes(raw)
                subprocess.run([str(driver), str(table)], check=True)
        # Many small compressed blocks exceed the old 64-entry cap and force
        # hash growth, collision chains, LRU eviction and close/reset cleanup.
        source.write_bytes(b"".join(i.to_bytes(2, "little") * 64 for i in range(300)))
        packed = pack_native(source, table, 128, 1, False)
        if packed["raw_blocks"]:
            raise AssertionError("cache sanitizer fixture must use compressed blocks")
        subprocess.run([str(driver), str(table)], check=True)
    print("ASan/UBSan: versions 1/2, 128 mutated/truncated files and 300-block cache growth/eviction checked.")


if __name__ == "__main__":
    main()
