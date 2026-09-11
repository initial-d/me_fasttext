#!/usr/bin/env python3
"""Prepare a public-corpus manifest and query file for me_fasttext benchmarks.

The helper does not download data, train a model, or inspect private files. It
turns a plain-text corpus into the metadata and query slices expected by
docs/first_public_corpus_benchmark.md.
"""

import argparse
import collections
import json
import os
import re
from pathlib import Path


TOKEN_RE = re.compile(r"\S+")
SLICE_DESCRIPTIONS = {
    "in_vocab": "Lines whose first tokens are all frequent in the corpus.",
    "oov_heavy": "Lines containing at least one token observed once.",
    "entity_heavy": "Lines containing digits, separators, or mixed-case entity-like tokens.",
}


def positive_int(value):
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a corpus manifest and query slices for a me_fasttext public benchmark."
    )
    parser.add_argument("--input", required=True, help="Plain-text corpus, one document or sentence per line.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated manifest and query files.")
    parser.add_argument("--name", required=True, help="Corpus name to record in the manifest.")
    parser.add_argument("--language", required=True, help="Corpus language, for example en, zh, multilingual.")
    parser.add_argument("--license", default="unknown", help="Corpus license or redistribution note.")
    parser.add_argument("--split", default="train", help="Split name represented by the input file.")
    parser.add_argument("--max-queries-per-slice", type=positive_int, default=5000)
    parser.add_argument("--min-line-tokens", type=positive_int, default=3)
    return parser.parse_args()


def tokenize(line):
    return TOKEN_RE.findall(line.strip())


def is_entity_like(token):
    if any(ch.isdigit() for ch in token):
        return True
    if "-" in token or "_" in token or "/" in token:
        return True
    letters = [ch for ch in token if ch.isalpha()]
    return bool(letters) and any(ch.isupper() for ch in letters[1:])


def add_limited(bucket, value, limit):
    if len(bucket) < limit and value not in bucket:
        bucket.append(value)


def prepare_public_benchmark(
    input_path,
    output_dir,
    name,
    language,
    license_note="unknown",
    split="train",
    max_queries_per_slice=5000,
    min_line_tokens=3,
):
    if max_queries_per_slice <= 0:
        raise ValueError("max_queries_per_slice must be a positive integer")
    if min_line_tokens <= 0:
        raise ValueError("min_line_tokens must be a positive integer")

    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"input file not found: {input_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    token_counts = collections.Counter()
    lines = []
    total_tokens = 0

    with input_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            tokens = tokenize(line)
            if len(tokens) < min_line_tokens:
                continue
            lines.append((line, tokens))
            token_counts.update(tokens)
            total_tokens += len(tokens)

    rare_tokens = {token for token, count in token_counts.items() if count == 1}
    common_tokens = {token for token, count in token_counts.items() if count >= 5}

    slices = {
        "in_vocab": [],
        "oov_heavy": [],
        "entity_heavy": [],
    }

    for line, tokens in lines:
        if tokens and all(token in common_tokens for token in tokens[: min(5, len(tokens))]):
            add_limited(slices["in_vocab"], line, max_queries_per_slice)
        if any(token in rare_tokens for token in tokens):
            add_limited(slices["oov_heavy"], line, max_queries_per_slice)
        if any(is_entity_like(token) for token in tokens):
            add_limited(slices["entity_heavy"], line, max_queries_per_slice)

    slice_paths = {}
    for slice_name, slice_lines in slices.items():
        slice_path = output_dir / f"queries_{slice_name}.txt"
        slice_paths[slice_name] = slice_path
        with slice_path.open("w", encoding="utf-8", newline="\n") as handle:
            for line in slice_lines:
                handle.write(line + "\n")

    query_path = output_dir / "queries.txt"
    merged_queries = []
    seen_queries = set()
    for slice_lines in slices.values():
        for line in slice_lines:
            if line not in seen_queries:
                merged_queries.append(line)
                seen_queries.add(line)
    with query_path.open("w", encoding="utf-8", newline="\n") as handle:
        for line in merged_queries:
            handle.write(line + "\n")

    slice_manifest_path = output_dir / "query_slices.md"
    manifest = {
        "corpus": {
            "name": name,
            "language": language,
            "license": license_note,
            "split": split,
            "source_path": os.path.basename(str(input_path)),
            "documents_or_lines": len(lines),
            "tokens": total_tokens,
            "unique_tokens": len(token_counts),
            "rare_tokens_count_1": len(rare_tokens),
            "min_line_tokens": min_line_tokens,
        },
        "query_slices": {
            slice_name: {
                "lines": len(slice_lines),
                "description": SLICE_DESCRIPTIONS[slice_name],
            }
            for slice_name, slice_lines in slices.items()
        },
        "outputs": {
            "queries": str(query_path.as_posix()),
            "slice_manifest": str(slice_manifest_path.as_posix()),
            "slice_queries": {
                name: str(path.as_posix()) for name, path in slice_paths.items()
            },
        },
    }

    manifest_path = output_dir / "corpus_manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    with slice_manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Query slices\n\n")
        handle.write(f"- Corpus: {name}\n")
        handle.write(f"- Language: {language}\n")
        handle.write(f"- License: {license_note}\n")
        handle.write(f"- Source split: {split}\n\n")
        handle.write("| Slice | Lines | Selection rule |\n")
        handle.write("| --- | ---: | --- |\n")
        for slice_name, meta in manifest["query_slices"].items():
            handle.write(f"| {slice_name} | {meta['lines']} | {meta['description']} |\n")

    return {
        "manifest": manifest_path,
        "queries": query_path,
        "slice_queries": slice_paths,
        "slice_manifest": slice_manifest_path,
    }


def main():
    args = parse_args()

    try:
        outputs = prepare_public_benchmark(
            args.input,
            args.output_dir,
            args.name,
            args.language,
            args.license,
            args.split,
            args.max_queries_per_slice,
            args.min_line_tokens,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(f"wrote {outputs['manifest']}")
    print(f"wrote {outputs['queries']}")
    for slice_path in outputs["slice_queries"].values():
        print(f"wrote {slice_path}")
    print(f"wrote {outputs['slice_manifest']}")


if __name__ == "__main__":
    main()
