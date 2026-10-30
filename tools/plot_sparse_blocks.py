"""Plot measured lossless storage versus sparse-query costs from a completed ablation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def summarize(report):
    if report['status'] != 'passed':
        raise ValueError('only complete measurements can be summarized')
    summary = {}
    for name, worker in report['workers'].items():
        packed = report['variants'].get(name)
        item = dict(artifact_bytes=report['raw_bytes'] if packed is None else packed['artifact_bytes'], cases={})
        item['saving_percent'] = 100 * (1 - item['artifact_bytes'] / report['raw_bytes'])
        for case in worker['result']['cases']:
            work = case['first_codec_work']
            decoded = None if not work or work[0] is None else sum(w['last_decoded_bytes'] for w in work)
            input_bytes = None if decoded is None else sum(w['last_compressed_bytes'] for w in work)
            mem = case['memory_after_first_trace']
            item['cases'][case['name']] = dict(
                first_p50_ms=case['first_trace']['p50_ms'], first_p95_ms=case['first_trace']['p95_ms'],
                hot_p50_ms=case['hot_repeated']['p50_ms'], first_cpu_p50_ms=case['first_cpu']['p50_ms'],
                rss_bytes=mem.get('Rss_bytes', mem['peak_rss_bytes']), anonymous_bytes=mem.get('Anonymous_bytes'),
                output_bytes=case['query_output_buffer_bytes'], decoded_bytes=decoded,
                compressed_input_bytes=input_bytes,
                decoded_to_output_ratio=None if decoded is None else decoded / case['query_output_buffer_bytes'])
        summary[name] = item
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(json.loads(args.report.read_text()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    names = list(summary)
    labels = [n.replace('raw-mmap', 'Raw mmap').replace('block-', '') + (' rows' if n.startswith('block') else '') for n in names]
    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].bar(x, [summary[n]['saving_percent'] for n in names], color='#2563eb')
    axes[0, 0].set_title('Lossless row-table saving')
    axes[0, 0].set_ylabel('Saving versus raw (%)')
    colors = ['#16a34a', '#d97706']
    for index, (case, label) in enumerate((('heldout-lexical', 'Held-out lexical'), ('uniform-id-stress', 'Uniform ID stress'))):
        for axis, metric in ((axes[0, 1], 'first_p50_ms'), (axes[1, 1], 'rss_bytes')):
            values = [summary[n]['cases'][case][metric] / (1024**2 if metric == 'rss_bytes' else 1) for n in names]
            axis.plot(x, values, 'o-', color=colors[index], label=label)
        values = [summary[n]['cases'][case]['decoded_to_output_ratio'] for n in names]
        axes[1, 0].plot(x, [np.nan if v is None else v for v in values], 'o-', color=colors[index], label=label)
    axes[0, 1].set_title('First trace, empty application cache')
    axes[0, 1].set_ylabel('512-row gather p50 (ms, log)')
    axes[0, 1].set_yscale('log')
    axes[1, 0].set_title('Full decoded blocks / requested output')
    axes[1, 0].set_ylabel('Logical decode amplification (x)')
    axes[1, 1].set_title('Whole-process RSS after first trace')
    axes[1, 1].set_ylabel('RSS including mapped pages (MiB)')
    for axis in axes.flat:
        axis.set_xticks(x, labels)
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
    for axis in (axes[0, 1], axes[1, 0], axes[1, 1]):
        axis.legend(fontsize=9)
    fig.suptitle('Same 1.85M trained rows: native sparse-block tradeoff', fontsize=15)
    fig.text(.5, .015, 'Byte-exact CPU lookup; shared host, warm filesystem. Uniform access is synthetic. Codec bytes are not disk I/O.\n'
             'RSS includes reclaimable file pages and outputs; no GPU or whole-model speedup measured.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .94))
    for extension in ('png', 'svg'):
        fig.savefig(args.output_dir / ('sparse-block-tradeoffs.' + extension), dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
