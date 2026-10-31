import json,sys,time
from pathlib import Path
root=Path('/llm-align/duyimin/ngram-scale-20261031')
sys.path.insert(0,str(root/'source-auto'))
from tools.benchmark_quantized_capacity import software_environment, memory
from tools.quantized_memory import pack_quantized_file
source=Path('/llm-align/duyimin/meft-cpu-capacity-20261030/training2/trained.f32')
report=dict(status='running',**software_environment(),memory_before=memory(),
            label='auto INT8 native/raw bundle selection',shape=[1855907,256])
started,cpu=time.perf_counter(),time.process_time()
metadata=pack_quantized_file(source,root/'auto-int8',256,'int8',native='auto',chunk_rows=4096,max_block_bytes=8192)
report.update(status='passed',metadata=metadata,wall_seconds=time.perf_counter()-started,
              cpu_seconds=time.process_time()-cpu,memory_after=memory(),
              bundle_bytes=sum(p.stat().st_size for p in (root/'auto-int8').iterdir()))
current=json.loads((root/'results/build-int8-raw.json').read_text())['metadata']
for key in ('source_float32_sha256','decoded_float32_sha256','table_sha256'):
 if current[key] != metadata[key]: raise AssertionError('auto/raw differs: '+key)
if metadata['backend'] != 'raw_mmap': raise AssertionError('expected measured INT8 expansion to select raw')
report['auto_matches_verified_raw_source_decoded_and_table_hashes']=True
(root/'auto-build.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('status','wall_seconds','cpu_seconds','memory_after','bundle_bytes','auto_matches_verified_raw_source_decoded_and_table_hashes')}))
