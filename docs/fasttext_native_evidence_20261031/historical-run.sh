set -eu
root=/llm-align/duyimin/meft-cpu-capacity-20261030/fusion-20261031
python=/llm-align/duyimin/meft-cuda-20261030/compat/bin/python
cd "$root/historical"
"$python" - <<'PY'
from pathlib import Path
from collections import Counter
import random
r=Path('/llm-align/duyimin/meft-cpu-capacity-20261030/fusion-20261031')
s=(r/'source/tools/fasttext_serving_probe.cc').read_text().replace('model.hasNativeInput()','false')
(r/'historical/probe.cc').write_text(s)
counts=Counter(token.split('\x01')[0] for token in (r/'reproduction/pairs.txt').read_text().split() if '\x01' in token)
known=sorted(w for w,c in counts.items() if c>=2)
rng=random.Random(42)
queries=['w\t'+rng.choice(known) for _ in range(1024)]+['s\t'+' '.join(rng.choices(known,k=8)) for _ in range(128)]
(r/'historical/known.tsv').write_text('\n'.join(queries)+'\n')
PY
c++ -std=c++11 -O2 -pthread -Isrc -Imylib/include probe.cc src/args.cc src/dictionary.cc src/productquantizer.cc src/matrix.cc src/qmatrix.cc src/vector.cc src/model.cc src/utils.cc src/fasttext.cc src/ftindex.cc -Lmylib/lib -ltrie -lpub -lz -o probe > build.txt 2>&1
./probe serve "$root/reproduction/trained.bin" known.tsv old.f32 old.json > old.txt 2>&1
"$root/reproduction/probe" serve "$root/reproduction/trained.bin" known.tsv dense.f32 dense.json > dense.txt 2>&1
"$root/reproduction/probe" serve "$root/reproduction/exact" known.tsv native.f32 native.json > native.txt 2>&1
"$python" - <<'PY'
import hashlib,json,platform,sys
from pathlib import Path
r=Path('/llm-align/duyimin/meft-cpu-capacity-20261030/fusion-20261031/historical')
hashes={x:hashlib.sha256((r/(x+'.f32')).read_bytes()).hexdigest() for x in ('old','dense','native')}
report={'historical_commit':'bea4302bb7fb1a6c0108fdde141ce667f2389920','fusion_commit':'bd6bd86eb38c5dec9ac63c35eb7ee3cfc895a2b8','query_sha256':hashlib.sha256((r/'known.tsv').read_bytes()).hexdigest(),'oracle_sha256':hashes,'byte_equal':len(set(hashes.values()))==1,'word_queries':1024,'sentence_queries':128,'query_seed':42,'python':sys.version,'platform':platform.platform(),'limitations':['Known registered words only. Prior OOV insertion and hash-bucket helper behavior can address untrained/out-of-bounds rows and is deliberately fixed.','Same actual trained model; source-compatible facade comparison, not binary ABI compatibility.','Historical probe only substitutes false for new hasNativeInput method; dense math and loader are original source.']}
(r/'report.json').write_text(json.dumps(report,indent=2)+'\n')
assert report['byte_equal'],report
print(json.dumps(report,indent=2))
PY
