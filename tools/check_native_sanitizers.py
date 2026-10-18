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
        source.write_bytes(b"".join(bytes([i % 13]) * 256 for i in range(128)))
        pack_native(source, table, 256, 8)
        valid = table.read_bytes()
        rng = random.Random(7)
        subprocess.run([str(driver), str(table)], check=True)
        for i in range(64):
            raw = bytearray(valid)
            for _ in range(1 + i % 3):
                raw[rng.randrange(len(raw))] ^= rng.randrange(1, 256)
            if i % 4 == 0:
                raw = raw[:rng.randrange(len(raw))]
            table.write_bytes(raw)
            subprocess.run([str(driver), str(table)], check=True)
    print("ASan/UBSan: valid artifact and 64 mutated/truncated files checked.")


if __name__ == "__main__":
    main()
