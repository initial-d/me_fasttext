"""Export a dense me_fasttext .bin into a frozen native FastText serving bundle.

Uses this repository's Linux C++ loader and original DATrie/mark-compact bridge.
Official fastText .bin layouts, quantized models and legacy .z are separate.
Build-time rows/maps may be table-sized; reader caches do not bound export RSS.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile

from .native_row_store import NativeRows
from .structural_native import StructuralPlan


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024**2), b''):
            digest.update(part)
    return digest.hexdigest()


def read_keys(path, rows):
    keys = {}
    with Path(path).open('rb') as stream:
        if stream.read(8) != b'MEFTKEY1':
            raise ValueError('invalid extracted key registry')
        count_bytes = stream.read(8)
        if len(count_bytes) != 8:
            raise ValueError('truncated key registry')
        count, = struct.unpack('<Q', count_bytes)
        if count > rows:
            raise ValueError('key count exceeds original rows')
        for _ in range(count):
            header = stream.read(16)
            if len(header) != 16:
                raise ValueError('truncated key record')
            row_id, size = struct.unpack('<QQ', header)
            if row_id >= rows or row_id in keys or size == 0 or size > path.stat().st_size:
                raise ValueError('invalid key record')
            raw = stream.read(size)
            if len(raw) != size or b'\0' in raw:
                raise ValueError('truncated or NUL-containing key')
            keys[row_id] = raw.decode('utf-8')
        if stream.read(1) or len(set(keys.values())) != count:
            raise ValueError('trailing bytes or duplicate keys')
    return keys


def export_fasttext_model(model, output_dir, *, fasttext_binary, structural_library,
                         native_library=None, mode='exact', cosine=.99, norm_ratio=.99,
                         max_block_bytes=8192, cache_bytes=4*1024**2, query_chunk_rows=256):
    """Publish a new owned bundle; source model and legacy file paths stay intact."""
    model, output = Path(model).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError('use a fresh native bundle directory')
    if mode not in ('layout', 'exact', 'joint'):
        raise ValueError('mode must be layout, exact or joint')
    if type(cache_bytes) is not int or cache_bytes < 0 or type(query_chunk_rows) is not int or query_chunk_rows <= 0:
        raise ValueError('cache budget must be nonnegative and query chunk rows positive')
    output.parent.mkdir(parents=True, exist_ok=True)
    original_hash = sha256(model)
    with tempfile.TemporaryDirectory(prefix='.meft-fasttext-', dir=output.parent) as tmp:
        extracted = Path(tmp)/'extracted'
        command = [str(Path(fasttext_binary).resolve()), 'dump-native', str(model), str(extracted)]
        run = subprocess.run(command, text=True, capture_output=True)
        if run.returncode:
            raise RuntimeError('dense model extraction failed:\n'+run.stdout+run.stderr)
        metadata = extracted/'model.meta'
        with metadata.open('rb') as stream:
            if stream.read(8) != b'MEFTFT01':
                raise ValueError('unsupported native FastText metadata')
            rows, dim = struct.unpack('<QQ', stream.read(16))
        source = extracted/'input.f32'
        if source.stat().st_size != rows*dim*4:
            raise ValueError('extracted input rows differ from declared model shape')
        keys = read_keys(extracted/'keys.bin', rows)
        bundle = Path(tmp)/'bundle'
        with StructuralPlan(source.read_bytes(), dim, keys, structural_library,
                            mode=mode, cosine=cosine, norm_ratio=norm_ratio) as plan:
            storage = plan.export(source, bundle, max_block_bytes=max_block_bytes)
        shutil.copyfile(metadata, bundle/'model.meta')
        (bundle/'reader.cfg').write_bytes(b'MEFTCFG1'+struct.pack('<4Q', cache_bytes, query_chunk_rows, 1,
                                                              ('layout','exact','joint').index(mode)))
        digest = hashlib.sha256()
        with NativeRows(bundle/'table.men', native_library) as reader:
            for begin in range(0, rows, 4096):
                digest.update(reader.rows(range(begin, min(begin+4096, rows))))
        reconstructed = digest.hexdigest()
        if reconstructed != plan.reconstructed_sha256:
            raise AssertionError('native full-row reconstruction differs from structural representatives')
        if mode != 'joint' and reconstructed != sha256(source):
            raise AssertionError('lossless export changed original row bytes')
        if sha256(model) != original_hash:
            raise ValueError('dense source model changed during export')
        manifest = dict(schema_version=1, model_type='me_fasttext-native',
            shape=[rows,dim], keyed_ngram_rows=len(keys), mode=mode,
            source_model_sha256=original_hash, source_rows_sha256=sha256(source),
            reconstructed_rows_sha256=reconstructed, original_bytes_preserved=storage['original_bytes_preserved'],
            extraction_command=command, storage=storage, query_chunk_rows=query_chunk_rows,
            cache_payload_budget_bytes=cache_bytes,
            contract='Original DATrie registry IDs; original float32 ordered accumulation. Joint mode can change source weights.',
            limitations=['Trusted self-produced, Linux x86-64 legacy metadata/index serialization.',
                         'Frozen inference; output matrix/dictionary remain resident. Cache is not an RSS limit.',
                         'Not an official fastText converter or quantized/.z migration.'],
            artifacts={p.name:sha256(p) for p in bundle.iterdir() if p.is_file()})
        (bundle/'fasttext.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (bundle/'extraction.log').write_text(run.stdout+run.stderr)
        if output.exists():
            raise FileExistsError('native bundle appeared during export')
        os.rename(bundle,output)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--fasttext-binary',type=Path,required=True)
    parser.add_argument('--structural-library',type=Path,required=True)
    parser.add_argument('--native-library',type=Path)
    parser.add_argument('--mode',choices=('layout','exact','joint'),default='exact')
    parser.add_argument('--cosine',type=float,default=.99)
    parser.add_argument('--norm-ratio',type=float,default=.99)
    parser.add_argument('--max-block-bytes',type=int,default=8192)
    parser.add_argument('--cache-bytes',type=int,default=4*1024**2)
    parser.add_argument('--query-chunk-rows',type=int,default=256)
    args=parser.parse_args()
    result=export_fasttext_model(args.model,args.output_dir,fasttext_binary=args.fasttext_binary,
        structural_library=args.structural_library,native_library=args.native_library,mode=args.mode,
        cosine=args.cosine,norm_ratio=args.norm_ratio,max_block_bytes=args.max_block_bytes,
        cache_bytes=args.cache_bytes,query_chunk_rows=args.query_chunk_rows)
    print(json.dumps({'output_dir':str(args.output_dir.resolve()),'shape':result['shape'],
                      'original_bytes_preserved':result['original_bytes_preserved']},indent=2))


if __name__=='__main__':
    main()
