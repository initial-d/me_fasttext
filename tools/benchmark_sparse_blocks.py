"""CPU-only block-layout ablation on an unchanged trained table and query oracle.

Reuses capacity-study training/queries, verifies every original ID in a separate
phase, then runs fresh-process direct-output workers. Does not retrain/download.
Codec byte counters measure logical decode work, not physical filesystem I/O.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .benchmark_memory_capacity import environment
from .native_row_store import DESC, HEADER, NativeRows, pack_native
from .prepare_public_benchmark import file_sha256


def describe_table(path):
    with path.open('rb') as stream:
        magic, version, rows, width, unique, block_rows, count = HEADER.unpack(stream.read(HEADER.size))
        if magic != b'MEFTNAT1' or version not in (1, 2):
            raise ValueError('invalid table header')
        stream.seek(HEADER.size + rows * 8)
        codecs = [0, 0, 0]
        for _ in range(count):
            _, _, _, codec = DESC.unpack(stream.read(DESC.size))
            codecs[codec] += 1
    return dict(n_rows=rows, row_bytes=width, unique_rows=unique, block_rows=block_rows,
                blocks=count, artifact_bytes=path.stat().st_size, raw_blocks=codecs[0],
                zlib_blocks=codecs[1], xor_blocks=codecs[2])


def verify_all_rows(path, library, rows, width, expected_sha256):
    digest = hashlib.sha256()
    with NativeRows(path, library) as table:
        if table.header != {'n_rows': rows, 'row_bytes': width}:
            raise AssertionError('native table shape differs from source')
        for start in range(0, rows, 4096):
            digest.update(table.rows(range(start, min(start + 4096, rows))))
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise AssertionError('full original-ID reconstruction differs from unchanged trained bytes')
    return actual


def run(args):
    source = args.training_dir / 'trained.f32'
    manifest = json.loads(args.query_manifest.read_text())
    rows, dim = manifest['shape']
    width = dim * 4
    if source.stat().st_size != rows * width:
        raise ValueError('query oracle shape differs from source')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report_path = args.output_dir / 'report.json'
    report = dict(status='running', **environment(), raw_bytes=source.stat().st_size,
                  source_sha256=file_sha256(source), query_manifest_sha256=file_sha256(args.query_manifest),
                  native_library_sha256=file_sha256(args.native_library), shape=[rows, dim],
                  training=json.loads((args.training_dir / 'training.json').read_text()), variants={}, workers={},
                  limitations=['Shared CPU; fixed worker order; filesystem/page cache warm after verification.',
                               'Empty application cache on first trace; not disk-cold latency.',
                               'Uniform access is synthetic on genuine trained rows; no whole-model inference.',
                               'Decoded cache budget excludes mmap pages, output, scratch and runtime metadata.',
                               'No GPU, cgroup memory pressure, task quality or monetary cost measurement.'])

    def save():
        report_path.write_text(json.dumps(report, indent=2) + '\n')

    def worker(label, directory, backend):
        shutil.copyfile(args.query_manifest, directory / 'queries-ordinary.json')
        target = directory / 'worker.json'
        command = [sys.executable, '-m', 'tools.benchmark_memory_capacity',
                   '--training-dir', str(args.training_dir), '--output-dir', str(directory),
                   '--native-library', str(args.native_library), '--cache-bytes', str(args.cache_bytes),
                   '--repeat', str(args.repeat), '--worker', backend, '--worker-report', str(target)]
        result = subprocess.run(command, capture_output=True, text=True)
        (directory / 'worker.log').write_text(result.stdout + result.stderr)
        report['workers'][label] = dict(command=command, returncode=result.returncode)
        if target.exists():
            report['workers'][label]['result'] = json.loads(target.read_text())
        save()
        if result.returncode:
            raise RuntimeError('worker failed: ' + label)

    try:
        save()
        control = args.output_dir / 'raw-mmap'
        control.mkdir()
        worker('raw-mmap', control, 'raw-mmap')
        for block_rows in args.block_rows:
            label = 'block-' + str(block_rows)
            directory = args.output_dir / label
            directory.mkdir()
            table_path = directory / 'native-ordinary.men'
            started = time.perf_counter()
            reused = args.reuse_baseline is not None and block_rows == 256
            if reused:
                table_path.symlink_to(args.reuse_baseline.resolve())
            else:
                pack_native(source, table_path, width, block_rows, False, 'xor')
            packed = describe_table(table_path)
            if packed['block_rows'] != block_rows:
                raise AssertionError('requested layout differs from packed layout')
            packed.update(reused_artifact=reused, pack_seconds=None if reused else time.perf_counter() - started,
                          artifact_sha256=file_sha256(table_path))
            started = time.perf_counter()
            packed['reconstructed_sha256'] = verify_all_rows(
                table_path, args.native_library, rows, width, report['source_sha256'])
            packed['verify_seconds'] = time.perf_counter() - started
            packed['all_original_ids_byte_exact'] = True
            report['variants'][label] = packed
            save()
            print(json.dumps(dict(phase='verified', variant=label, **packed)), flush=True)
            worker(label, directory, 'native-ordinary')
            print(json.dumps(dict(phase='worker-passed', variant=label)), flush=True)
        report['status'] = 'passed'
        save()
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-dir', required=True, type=Path)
    parser.add_argument('--query-manifest', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--native-library', required=True, type=Path)
    parser.add_argument('--block-rows', nargs='+', type=int, default=[8, 32, 256])
    parser.add_argument('--reuse-baseline', type=Path, help='Existing 256-row artifact, fully verified again')
    parser.add_argument('--cache-bytes', type=int, default=4 * 1024**2)
    parser.add_argument('--repeat', type=int, default=20)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('set CUDA_VISIBLE_DEVICES to empty')
    if any(n <= 0 for n in args.block_rows) or len(set(args.block_rows)) != len(args.block_rows):
        parser.error('block rows must be distinct positive integers')
    if args.cache_bytes < 0 or args.repeat <= 0:
        parser.error('cache bytes must be nonnegative and repeat positive')
    run(args)


if __name__ == '__main__':
    main()
