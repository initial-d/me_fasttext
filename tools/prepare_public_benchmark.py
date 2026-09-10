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
    parser.add_argument("--max-queries-per-slice", type=int, default=5000)
    parser.add_argument("--min-line-tokens", type=int, default=3)
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


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
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
            if len(tokens) < args.min_line_tokens:
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
            add_limited(slices["in_vocab"], line, args.max_queries_per_slice)
        if any(token in rare_tokens for token in tokens):
            add_limited(slices["oov_heavy"], line, args.max_queries_per_slice)
        if any(is_entity_like(token) for token in tokens):
            add_limited(slices["entity_heavy"], line, args.max_queries_per_slice)

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
            "name": args.name,
            "language": args.language,
            "license": args.license,
            "split": args.split,
            "source_path": os.path.basename(str(input_path)),
            "documents_or_lines": len(lines),
            "tokens": total_tokens,
            "unique_tokens": len(token_counts),
            "rare_tokens_count_1": len(rare_tokens),
            "min_line_tokens": args.min_line_tokens,
        },
        "query_slices": {
            name: {
                "lines": len(slice_lines),
                "description": {
                    "in_vocab": "Lines whose first tokens are all frequent in the corpus.",
                    "oov_heavy": "Lines containing at least one token observed once.",
                    "entity_heavy": "Lines containing digits, separators, or mixed-case entity-like tokens.",
                }[name],
            }
            for name, slice_lines in slices.items()
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
        handle.write(f"- Corpus: {args.name}\n")
        handle.write(f"- Language: {args.language}\n")
        handle.write(f"- License: {args.license}\n")
        handle.write(f"- Source split: {args.split}\n\n")
        handle.write("| Slice | Lines | Selection rule |\n")
        handle.write("| --- | ---: | --- |\n")
        for name, meta in manifest["query_slices"].items():
            handle.write(f"| {name} | {meta['lines']} | {meta['description']} |\n")

    print(f"wrote {manifest_path}")
    print(f"wrote {query_path}")
    for slice_path in slice_paths.values():
        print(f"wrote {slice_path}")
    print(f"wrote {slice_manifest_path}")


if __name__ == "__main__":
    main()
