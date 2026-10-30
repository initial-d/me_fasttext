"""Plot recorded storage, whole-process RSS and CPU gather costs together."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if report['status'] != 'passed':
        raise ValueError('only completed evidence can be plotted')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    names = list(report['workers'])
    labels = {'resident':'Resident','raw-mmap':'Raw mmap','native-ordinary':'Native','native-exact':'DATrie exact','native-joint':'DATrie joint .99'}
    colors = ['#64748b','#94a3b8','#2563eb','#16a34a','#d97706']
    mib = 1024**2
    fig, axes = plt.subplots(1,4,figsize=(17,4.8))
    x = np.arange(len(names))
    storage = [report['raw_table_bytes'] if n not in report['variants'] else report['variants'][n]['storage']['artifact_bytes'] for n in names]
    axes[0].bar(x,np.asarray(storage)/mib,color=colors)
    axes[0].set_ylabel('Row-table artifact (MiB)')
    axes[0].set_title('Exact bytes / explicit approximation')
    rss = []
    anonymous = []
    hot = []
    first = []
    for name in names:
        worker = report['workers'][name]['result']
        case = next(c for c in worker['cases'] if c['name']=='heldout-lexical')
        rss.append(case['memory_after_hot'].get('Rss_bytes',case['memory_after_hot']['peak_rss_bytes'])/mib)
        anonymous.append(case['memory_after_hot'].get('Anonymous_bytes',0)/mib)
        hot.append(case['hot_repeated']['p50_ms'])
        first.append(case['first_trace']['p50_ms'])
    axes[1].bar(x,rss,color=colors)
    axes[1].set_ylabel('Whole-process RSS (MiB)')
    axes[1].set_title('After lexical replay + hot queries')
    axes[2].bar(x,anonymous,color=colors)
    axes[2].set_ylabel('Anonymous memory (MiB)')
    axes[2].set_title('Separate reclaimable file pages')
    axes[3].plot(x,first,'s',color='#d97706',label='First trace')
    axes[3].plot(x,hot,'o',color='#2563eb',label='Hot repeated')
    axes[3].set_yscale('log')
    axes[3].set_ylabel('Gather p50 (ms, log scale)')
    axes[3].set_title('512-row CPU query costs')
    axes[3].set_ylim(.01,max(first)*2)
    axes[3].legend(fontsize=8)
    for axis in axes:
        axis.set_xticks(x,[labels[n] for n in names],rotation=22,ha='right')
        axis.grid(axis='y',alpha=.2)
        axis.set_axisbelow(True)
    fig.suptitle('Trained n-gram memory: capacity and query tradeoffs',fontsize=14)
    fig.text(.5,.01,'One finite SGNS reference; warm shared CPU. RSS includes mapped pages. Joint .99 changes vectors; no task-quality or GPU claim.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.94))
    for extension in ('svg','png'):
        fig.savefig(args.output_dir/('capacity-tradeoffs.'+extension),dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
