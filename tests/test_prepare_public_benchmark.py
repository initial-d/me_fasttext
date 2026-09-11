import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "prepare_public_benchmark.py"


def load_tool_module():
    spec = importlib.util.spec_from_file_location("prepare_public_benchmark", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PreparePublicBenchmarkTest(unittest.TestCase):
    def test_generates_manifest_and_query_slices(self):
        tool = load_tool_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            corpus_path = tmp_path / "corpus.txt"
            output_dir = tmp_path / "out"
            corpus_path.write_text(
                "\n".join(
                    [
                        "common token shared bridge stable",
                        "common token shared bridge stable",
                        "common token shared bridge stable",
                        "common token shared bridge stable",
                        "common token shared bridge stable",
                        "rarealpha common token shared stable",
                        "ACME-42 common token shared stable",
                        "MixedCase common token shared stable",
                        "tiny",
                        "",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            outputs = tool.prepare_public_benchmark(
                corpus_path,
                output_dir,
                name="sample-corpus",
                language="en",
                license_note="CC0",
                max_queries_per_slice=2,
            )

            manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["corpus"]["documents_or_lines"], 8)
            self.assertEqual(manifest["corpus"]["tokens"], 40)
            self.assertEqual(manifest["corpus"]["source_path"], "corpus.txt")
            self.assertEqual(manifest["query_slices"]["in_vocab"]["lines"], 1)
            self.assertEqual(manifest["query_slices"]["oov_heavy"]["lines"], 2)
            self.assertEqual(manifest["query_slices"]["entity_heavy"]["lines"], 2)

            merged_queries = outputs["queries"].read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(merged_queries), len(set(merged_queries)))
            self.assertIn("common token shared bridge stable", merged_queries)
            self.assertIn("rarealpha common token shared stable", merged_queries)
            self.assertIn("ACME-42 common token shared stable", merged_queries)

            query_slices = {
                name: path.read_text(encoding="utf-8").splitlines()
                for name, path in outputs["slice_queries"].items()
            }
            self.assertEqual(query_slices["in_vocab"], ["common token shared bridge stable"])
            self.assertEqual(len(query_slices["oov_heavy"]), 2)
            self.assertEqual(len(query_slices["entity_heavy"]), 2)

    def test_rejects_non_positive_limits(self):
        tool = load_tool_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_path = Path(tmpdir) / "corpus.txt"
            corpus_path.write_text("one two three\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                tool.prepare_public_benchmark(
                    corpus_path,
                    Path(tmpdir) / "out",
                    name="sample",
                    language="en",
                    max_queries_per_slice=0,
                )

    def test_cli_reports_missing_input_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--input",
                    str(Path(tmpdir) / "missing.txt"),
                    "--output-dir",
                    str(Path(tmpdir) / "out"),
                    "--name",
                    "missing",
                    "--language",
                    "en",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("error: input file not found", result.stderr)
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
