import json,sys,time
from pathlib import Path
root=Path('/llm-align/duyimin/ngram-scale-20261031')
sys.path.insert(0,str(root/'source-old'))
import numpy as np
from tools.benchmark_memory_capacity import environment,memory
from tools.quantized_memory import pack_quantized
source=Path('/llm-align/duyimin/meft-cpu-capacity-20261030/training2/trained.f32')
shape=(1855907,256)
report=dict(status='running',**environment(),memory_before=memory(),
            label='pre-streaming main INT8 native export',shape=list(shape))
rows=np.memmap(source,mode='r',dtype='<f4',shape=shape)
started,cpu=time.perf_counter(),time.process_time()
metadata=pack_quantized(rows,root/'old-int8-native','int8',deduplicate=False,max_block_bytes=8192)
report.update(status='passed',metadata=metadata,wall_seconds=time.perf_counter()-started,
              cpu_seconds=time.process_time()-cpu,memory_after=memory())
current=json.loads((root/'results/build-int8-native.json').read_text())['metadata']
for key in ('source_float32_sha256','decoded_float32_sha256','table_sha256'):
 if current[key] != metadata[key]: raise AssertionError('old/new export differs: '+key)
report['old_new_source_decoded_and_table_hashes_equal']=True
(root/'old-build.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('status','wall_seconds','cpu_seconds','memory_after','old_new_source_decoded_and_table_hashes_equal')}))
