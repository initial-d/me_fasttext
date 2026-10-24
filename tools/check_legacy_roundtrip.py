"""Build and check the real legacy exporter/reader with bundled Linux libraries.

Only deterministic correctness fixtures are used. No training or data download.
The bundled archives target Linux x86-64; portable callback tests remain separate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["args", "dictionary", "productquantizer", "matrix", "qmatrix", "vector",
           "model", "utils", "fasttext", "ftindex"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        parser.error("bundled legacy libraries require Linux x86-64; run this check in Linux CI")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    executable = output / "legacy-roundtrip"
    compiler = os.environ.get("CXX", "c++")
    command = [compiler, "-std=c++11", "-O1", "-g", "-pthread",
               "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
               "-I" + str(ROOT / "src"), "-I" + str(ROOT / "mylib/include"),
               str(ROOT / "tests/legacy_roundtrip_test.cc"),
               *[str(ROOT / f"src/{name}.cc") for name in SOURCES],
               "-L" + str(ROOT / "mylib/lib"), "-ltrie", "-lpub", "-o", str(executable)]
    report = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "exact_command": shlex.join([sys.executable, *sys.argv]),
        "compile_command": shlex.join(command),
        "run_command": shlex.join([str(executable), str(output)]),
        "python": sys.version, "platform": platform.platform(), "cpu": platform.processor(),
        "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
        "category": "deterministic synthetic correctness / serialization / serving regression",
        "data_source": "hard-coded mathematical row and vocabulary fixtures; no external data",
        "dimensions": [64, 128, 256], "export_overloads": 2, "device": "CPU",
        "sanitizers": ["AddressSanitizer", "UndefinedBehaviorSanitizer", "LeakSanitizer"],
        "limitations": ["Not trained rows or a paper/Engram quality/compression benchmark.",
                        "Bundled trie archives are linked but not sanitizer-instrumented.",
                        "Tests valid self-produced files, not malformed-file reader hardening.",
                        "Checks word/OOV/ngram APIs; legacy sentence helpers still assume dimension 128."],
        "status": "running",
    }
    if Path("/proc/cpuinfo").exists():
        report["cpu"] = next((line.split(":", 1)[1].strip()
                              for line in Path("/proc/cpuinfo").read_text().splitlines()
                              if line.startswith("model name")), report["cpu"])
    report["library_sha256"] = {
        name: hashlib.sha256((ROOT / "mylib/lib" / name).read_bytes()).hexdigest()
        for name in ("libtrie.a", "libpub.a")
    }
    try:
        build = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (output / "build.txt").write_text(build.stdout)
        report["build_exit_code"] = build.returncode
        build.check_returncode()
        environment = dict(os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1",
                           UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
        result = subprocess.run([str(executable), str(output)], env=environment, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (output / "run.txt").write_text(result.stdout)
        report["run_exit_code"] = result.returncode
        print(result.stdout, end="")
        result.check_returncode()
        report["status"] = "passed"
        report["artifact_sha256"] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.glob("*.z")
        }
    except subprocess.CalledProcessError:
        report["status"] = "failed"
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
