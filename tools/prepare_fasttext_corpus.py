"""Convert plain UTF-8 text into the existing positional/source SGNS input.

Adjacent words stay in their original line and order. This is the custom
me_fasttext training protocol, not ordinary upstream fastText text training.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024**2),b''):digest.update(part)
    return digest.hexdigest()


def prepare_corpus(source, output, *, words_per_line=64, lowercase=False):
    source,output=Path(source).resolve(),Path(output).resolve()
    if type(words_per_line) is not int or words_per_line<2:
        raise ValueError('words_per_line must be at least two')
    if source==output:raise ValueError('source and output must differ')
    if output.exists():raise FileExistsError('use a fresh prepared-corpus path')
    output.parent.mkdir(parents=True,exist_ok=True)
    source_hash=sha256(source); count=lines=skipped=0; owned=False
    try:
        with output.open('x',encoding='utf-8',newline='\n') as dest:
            owned=True
            with source.open(encoding='utf-8') as text:
                for line in text:
                    words=(line.lower() if lowercase else line).split()
                    if any('\x01' in w or '\0' in w or w.startswith(('p#','s#','__label__')) or w=='</s>' for w in words):
                        raise ValueError('plain text contains reserved positional/source/label tokens')
                    for begin in range(0,len(words),words_per_line):
                        block=words[begin:begin+words_per_line]
                        if len(block)<2:skipped+=len(block); continue
                        pairs=[a+'\x01'+b for a,b in zip(block,block[1:])]
                        dest.write('p#0 '+' '.join(pairs)+'\n'); count+=len(pairs); lines+=1
        if not count:raise ValueError('need at least two words on one input line')
        if sha256(source)!=source_hash:raise ValueError('source changed during preparation')
    except BaseException:
        if owned:output.unlink(missing_ok=True)
        raise
    return {'schema_version':1,'source_sha256':source_hash,'prepared_sha256':sha256(output),
        'pair_tokens':count,'prepared_lines':lines,'skipped_singletons':skipped,
        'words_per_line':words_per_line,'lowercase':lowercase,
        'protocol':'Within-line adjacent query\\x01source pairs, prefixed p#0; custom positional/source SGNS.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--words-per-line',type=int,default=64)
    parser.add_argument('--lowercase',action='store_true')
    args=parser.parse_args()
    try:result=prepare_corpus(args.input,args.output,words_per_line=args.words_per_line,lowercase=args.lowercase)
    except (ValueError,OSError) as exc:parser.error(str(exc))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
