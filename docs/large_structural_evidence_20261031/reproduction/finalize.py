import json,os,signal,subprocess,time
from pathlib import Path
root=Path('/llm-align/duyimin/ngram-structure-20261031')
primary=root/'results/report.json'
while True:
    if primary.exists():
        report=json.loads(primary.read_text())
        if len([v for v in report['variants'] if v['encoding']=='fp32'])==9:break
    if (root/'status').read_text().strip()=='failed':raise RuntimeError('primary worker failed')
    time.sleep(10)
# Stop only this task's original CPU controller/children after its complete FP32
# slice, avoiding duplicate FP16/INT8 work. Preserve any partial extra artifacts.
controller=897893
def args(pid):
    try:return Path(f'/proc/{pid}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
    except FileNotFoundError:return []
def ours(pid):
    a=args(pid)
    return ('tools.benchmark_large_structure' in a and '--output-dir' in a
            and a[a.index('--output-dir')+1]==str(root/'results'))
children=subprocess.check_output(['ps','-o','pid=','--ppid',str(controller)],text=True).split()
if ours(controller):
    os.kill(controller,signal.SIGSTOP)
    for pid in map(int,children):
        if ours(pid):os.kill(pid,signal.SIGTERM)
    os.kill(controller,signal.SIGTERM);os.kill(controller,signal.SIGCONT)
time.sleep(2)
(root/'primary-at-handoff.json').write_text(json.dumps(report,indent=2)+'\n')
report['status']='controller-stopped-after-complete-fp32-slice'
report['orchestration']='Complete FP32 builds/verification retained; duplicate remaining encodings use separate CPU runs. Partial extra artifacts retained.'
primary.write_text(json.dumps(report,indent=2)+'\n')
(root/'status').write_text('fp32-slice-completed\n')
watcher=root/'query-launcher.pid'
if watcher.exists():
    pid=int(watcher.read_text())
    if str(root/'query-run.sh') in args(pid):os.kill(pid,signal.SIGTERM)
(root/'query-status').write_text('superseded-by-encoding-parallel-finalizer\n')
for encoding in ['fp16','int8']:
    while (root/('status-'+encoding)).read_text().strip()=='running':time.sleep(10)
    if (root/('status-'+encoding)).read_text().strip()!='completed':raise RuntimeError(encoding+' failed')
python='/llm-align/duyimin/miniconda3/envs/gemma4/bin/python'
subprocess.run([python,'-m','tools.merge_large_structure_reports',
    '--part','fp32='+str(primary),
    '--part','fp16='+str(root/'results-fp16/report.json'),
    '--part','int8='+str(root/'results-int8/report.json'),
    '--output',str(root/'merged-builds.json')],cwd=root/'final-source',check=True)
subprocess.run([python,'-m','tools.replay_large_structure_queries',
    '--report',str(root/'merged-builds.json'),'--output-dir',str(root/'query-final')],cwd=root/'final-source',check=True)
subprocess.run([python,'-m','tools.analyze_large_structure',
    '--report',str(root/'query-final/report.json'),'--output-dir',str(root/'analysis')],cwd=root/'final-source',check=True)
