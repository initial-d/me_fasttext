import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LegacyMergeTest(unittest.TestCase):
    def test_production_callbacks_direction_magnitude_structure_and_final_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "legacy-merge"
            subprocess.run([
                os.environ.get("CXX", "c++"), "-std=c++11", "-Wall", "-Wextra", "-Werror",
                "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g",
                "-I" + str(ROOT / "src"), "-I" + str(ROOT / "mylib/include"),
                str(ROOT / "tests/legacy_merge_test.cc"), "-o", str(executable),
            ], check=True)
            subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    unittest.main()
