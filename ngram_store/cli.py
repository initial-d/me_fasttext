"""NGramStore: pack, inspect and query frozen n-gram vector tables."""
import argparse
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
import struct
import sys
import tempfile

from . import NativeRows, pack_native

FORWARD={
    'prepare-fasttext':('prepare_fasttext_corpus','Prepare plain text for the custom FastText SGNS trainer.'),
    'fasttext-export':('fasttext_native_bundle','Export a dense legacy FastText model into the same native backend.'),
    'tiny-export':('tinyengram_checkpoint','Export a TinyEngram/Qwen checkpoint.'),
    'tiny-verify':('verify_tinyengram_checkpoint','Verify a converted TinyEngram checkpoint on CPU.'),
}


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024**2),b''):digest.update(part)
    return digest.hexdigest()


def pack_store(source, output, dim, *, native_library=None, keys=None, structural_library=None,
               mode='exact', cosine=.99, norm_ratio=.99, max_block_bytes=8192, deduplicate=False):
    if sys.byteorder!='little':raise ValueError('CLI raw input requires little-endian IEEE float32 host')
    source,output=Path(source).resolve(),Path(output).resolve()
    if type(dim) is not int or dim<=0:raise ValueError('dim must be positive')
    rows,tail=divmod(source.stat().st_size,dim*4)
    if not rows or tail:raise ValueError('input must contain nonempty complete float32 rows')
    if output.exists():raise FileExistsError('use a fresh store directory')
    if mode not in ('layout','exact','joint'):raise ValueError('invalid structural mode')
    if keys is None and mode=='joint':raise ValueError('joint sharing requires registered string keys')
    if keys is not None and deduplicate:raise ValueError('structural representatives and generic deduplication are separate modes')
    source_hash=sha256(source); output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.ngram-store-',dir=output.parent) as tmp:
        candidate=Path(tmp)/'bundle'
        if keys is not None:
            module=import_module('meft_memory._runtime.structural_native')
            with module.StructuralPlan(source.read_bytes(),dim,keys,structural_library,
                                       mode=mode,cosine=cosine,norm_ratio=norm_ratio) as plan:
                storage=plan.export(source,candidate,max_block_bytes=max_block_bytes)
                expected=plan.reconstructed_sha256; preserved=plan.original_bytes_preserved
        else:
            candidate.mkdir()
            storage=pack_native(source,candidate/'table.men',dim*4,256,deduplicate,predictor='xor',
                                max_block_bytes=max_block_bytes)
            expected=source_hash; preserved=True
        reconstructed=hashlib.sha256()
        with NativeRows(candidate/'table.men',native_library) as table:
            for begin in range(0,rows,4096):reconstructed.update(table.rows(range(begin,min(begin+4096,rows))))
        if reconstructed.hexdigest()!=expected:raise ValueError('full-row reconstruction differs from selected representatives')
        if preserved and expected!=source_hash:raise ValueError('lossless pack changed source bytes')
        if sha256(source)!=source_hash:raise ValueError('source changed during export')
        artifact_hash=sha256(candidate/'table.men')
        # The same artifact can be opened by the existing frozen PyTorch adapter.
        embedding={'schema_version':1,'table_file':'table.men','shape':[rows,dim],
            'dtype':'torch.float32','byteorder':'little','head_sizes':[rows],'offsets':[0],
            'backend':'native','weight_sha256':expected,'source_weight_sha256':source_hash,
            'artifact_sha256':artifact_hash,'approximate':not preserved,**storage}
        (candidate/'embedding.json').write_text(json.dumps(embedding,indent=2)+'\n')
        metadata={'schema_version':1,'engine':'NGramStore','shape':[rows,dim],'dtype':'float32-le',
            'source_sha256':source_hash,'reconstructed_sha256':expected,'artifact_sha256':artifact_hash,
            'original_bytes_preserved':preserved,'keyed_rows':len(keys) if keys is not None else 0,
            'structural_mode':mode if keys is not None else None,'storage':storage,
            'contract':'Original numeric IDs; decoder preserves representative bytes. Joint sharing can change source weights.',
            'limitations':['Frozen single-worker handles; cache budget is not an RSS bound.',
                           'Structural mode requires trusted Linux x86-64 DATrie artifacts and full offline planning.',
                           'Hash collisions do not certify prefix/suffix relations.']}
        (candidate/'store.json').write_text(json.dumps(metadata,indent=2)+'\n')
        if output.exists():raise FileExistsError('store directory appeared during export')
        os.rename(candidate,output)
    return metadata


def read_keys(path):
    raw=json.loads(Path(path).read_text())
    if not isinstance(raw,dict):raise ValueError('keys JSON must map original row IDs to UTF-8 strings')
    result={}
    for key,value in raw.items():
        if not key.isdigit() or str(int(key))!=key or not isinstance(value,str):
            raise ValueError('keys must use canonical nonnegative row IDs and string values')
        result[int(key)]=value
    return result


def open_metadata(path):
    root=Path(path)
    name='store.json' if (root/'store.json').is_file() else 'fasttext.json'
    metadata=json.loads((root/name).read_text())
    if metadata.get('schema_version')!=1:raise ValueError('unsupported store metadata')
    return metadata


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in FORWARD:
        name=argv[0]; old=sys.argv
        try:
            sys.argv=['ngram-store '+name,*argv[1:]]
            return import_module('meft_memory._runtime.'+FORWARD[name][0]).main()
        finally:sys.argv=old
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    pack=commands.add_parser('pack',help='Pack existing little-endian float32 rows and verify every reconstructed row.')
    pack.add_argument('--input',type=Path,required=True); pack.add_argument('--output',type=Path,required=True)
    pack.add_argument('--dim',type=int,required=True); pack.add_argument('--native-library',type=Path)
    pack.add_argument('--keys',type=Path); pack.add_argument('--structural-library',type=Path)
    pack.add_argument('--mode',choices=('layout','exact','joint'),default='exact')
    pack.add_argument('--cosine',type=float,default=.99); pack.add_argument('--norm-ratio',type=float,default=.99)
    pack.add_argument('--max-block-bytes',type=int,default=8192); pack.add_argument('--deduplicate',action='store_true')
    inspect=commands.add_parser('inspect',help='Inspect row shape, storage bytes and fidelity contract.')
    query=commands.add_parser('query',help='Query original row IDs or registered string keys.')
    for command in (inspect,query):
        command.add_argument('--store',type=Path,required=True); command.add_argument('--native-library',type=Path)
    selection=query.add_mutually_exclusive_group(required=True)
    selection.add_argument('--ids',type=int,nargs='+'); selection.add_argument('--keys',nargs='+')
    query.add_argument('--structural-library',type=Path); query.add_argument('--cache-bytes',type=int,default=4*1024**2)
    query.add_argument('--hex',action='store_true',help='Return original row bytes, including nonfinite/signed-zero bits.')
    for name,(_,help_text) in FORWARD.items():commands.add_parser(name,help=help_text,add_help=False)
    args=parser.parse_args(argv)
    try:
        if args.command=='pack':
            result=pack_store(args.input,args.output,args.dim,native_library=args.native_library,
                keys=read_keys(args.keys) if args.keys else None,structural_library=args.structural_library,
                mode=args.mode,cosine=args.cosine,norm_ratio=args.norm_ratio,
                max_block_bytes=args.max_block_bytes,deduplicate=args.deduplicate)
        else:
            metadata=open_metadata(args.store); rows,dim=metadata['shape']
            ids=None
            if args.command=='query':
                ids=args.ids
                if args.keys is not None:
                    module=import_module('meft_memory._runtime.structural_native')
                    with module.StructuralIndex(args.store/'keys.das',args.structural_library) as index:ids=index.lookup(args.keys)
                    if None in ids:raise ValueError('query contains an unregistered key')
            with NativeRows(args.store/'table.men',args.native_library,
                            args.cache_bytes if args.command=='query' else 0) as table:
                if table.header['n_rows']!=rows or table.header['row_bytes']!=dim*4:
                    raise ValueError('row-store shape differs from metadata')
                if ids is None:result=metadata
                else:
                    raw=table.rows(ids)
                    result={'ids':ids,'original_bytes_preserved':metadata['original_bytes_preserved']}
                    if args.hex:result['rows_hex']=[raw[i*dim*4:(i+1)*dim*4].hex() for i in range(len(ids))]
                    else:result['vectors']=[list(struct.unpack_from('<'+str(dim)+'f',raw,i*dim*4)) for i in range(len(ids))]
        print(json.dumps(result,indent=2,allow_nan=False))
    except (ValueError,OSError,KeyError,TypeError) as exc:parser.error(str(exc))


if __name__=='__main__':main()
