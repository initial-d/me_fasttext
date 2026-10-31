"""Summarize complete scale ablations; never promote incomplete workers."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def analyze(report):
    if report['status']!='passed' or any(w['returncode'] or w['result']['status']!='passed' for w in report['workers']):
        raise ValueError('incomplete or failed experiment; retain raw failure evidence')
    info=report['provenance']; rows=info['shape'][0]
    result=dict(source_sha256=info['source_sha256'],shape=info['shape'],
        source_commit=info['source_commit'],settings=info['settings'],limitations=info['limitations'],
        keys_bytes=info['keys_bytes'],index_bytes=report['index_bytes'],variants=[],pair_strata=[],queries={})
    result['parts']=report.get('parts',[])
    result['query_replay']=report.get('query_replay')
    result['worker_source_commits']=sorted({w['result']['source_commit'] for w in report['workers']})
    audit=next(w['result'] for w in report['workers'] if w['phase']=='audit')
    result['row_norm_by_order']=audit['by_order']
    for s in audit['strata']:
        actual=s['structural']; control=s['unrelated_length_boundary_matched']
        fixed=next(t for t in actual['thresholds'] if t['cosine']==.99)
        result['pair_strata'].append(dict(order=s['order'],target_order=s['target_order'],direction=s['direction'],
            source_population=s['source_population'],sampled_sources=s['sampled_sources'],pairs=actual['pairs'],
            cosine=actual['cosine'],norm_ratio=actual['norm_ratio'],fixed099=fixed,
            shared_affix_length=s.get('shared_affix_length'),control=control))
    control_bytes={}
    for encoding in info['settings']['encodings']:
        controls=[v for v in report['variants'] if v['encoding']==encoding and v['policy']!='joint']
        best=min(controls,key=lambda v:v['build']['bundle_bytes'])
        control_bytes[encoding]=best['build']['bundle_bytes']+info['keys_bytes']
    raw_fp32=next(v for v in report['variants'] if v['encoding']=='fp32' and v['policy']=='raw')
    raw_bytes=raw_fp32['build']['bundle_bytes']+info['keys_bytes']
    for v in report['variants']:
        verification=v['verification']
        if (verification['all_rows_verified']!=rows
            or not verification['all_encoded_and_decoded_original_ids_byte_exact_to_representative']):
            raise ValueError('full original-ID verification missing')
        total=v['build']['bundle_bytes']+info['keys_bytes']
        result['variants'].append(dict(name=v['name'],encoding=v['encoding'],policy=v['policy'],threshold=v['threshold'],
            bundle_bytes=v['build']['bundle_bytes'],bundle_with_common_keys_bytes=total,
            bundle_with_datrie_bytes=v['build']['bundle_bytes']+report['index_bytes']['keys.das'],
            saving_vs_raw_fp32=1-total/raw_bytes,
            additional_saving_vs_best_same_encoding_control=1-total/control_bytes[v['encoding']],
            best_same_encoding_control_bytes=control_bytes[v['encoding']],
            table_storage=v['build']['metadata']['storage'],build_seconds=v['build']['build_seconds'],
            build_peak_rss_bytes=v['build']['memory_after']['peak_rss_bytes'],
            fidelity=verification['overall'],merged_only=verification['merged_only'],by_order=verification['by_order'],
            query_weighted_relative_frobenius=verification['heldout_query_weighted_relative_frobenius'],
            query_weighted_added_frobenius=verification['heldout_query_weighted_added_frobenius'],
            composed_ngram_mean=verification['composed_ngram_mean']))
    for v in report['variants']:
        result['queries'][v['name']]={}
        for case in ('heldout-lexical','uniform-id-stress'):
            matches=[w['result'] for w in report['workers'] if w['phase']=='query'
                and w['name'].startswith('query-'+v['name']+'-'+case+'-')]
            if len(matches)!=info['settings']['rounds'] or not all(w['all_first_and_second_query_bytes_exact'] for w in matches):
                raise ValueError('missing independent query rounds')
            summary=dict(process_rounds=len(matches))
            for phase in ('first_trace','second_trace','hot_trace'):
                summary[phase]=dict(process_p50_ms=[w[phase]['wall']['p50_ms'] for w in matches],
                    median_process_p50_ms=float(np.median([w[phase]['wall']['p50_ms'] for w in matches])),
                    median_process_cpu_p50_ms=float(np.median([w[phase]['cpu']['p50_ms'] for w in matches])))
            for stage in ('memory_open','memory_first','memory_second','memory_hot'):
                summary[stage]={key:float(np.median([w[stage].get(key,0) for w in matches]))
                    for key in ('VmRSS_bytes','Anonymous_bytes','peak_rss_bytes')}
            result['queries'][v['name']][case]=summary
    return result


def plot_training_support(support, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    orders=sorted(map(int,support['by_order']))
    values=[support['by_order'][str(n)] for n in orders]
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    axes[0].plot(orders,[100*v['one_source_word_fraction'] for v in values],'o-')
    axes[0].set_ylabel('Rows supported by one source word (%)')
    axes[1].plot(orders,[v['training_update_relative_to_initial']['p50'] for v in values],'o-')
    axes[1].set_yscale('log');axes[1].set_ylabel('Median training update / initial norm')
    axes[2].plot(orders,[v['cosine_to_initial']['p50'] for v in values],'o-')
    axes[2].set_ylabel('Median cosine to actual initialization')
    for ax in axes:ax.set_xlabel('N-gram character length');ax.grid(alpha=.2)
    fig.suptitle('Potential word supports and actual updates; one finite CPU SGNS checkpoint, not convergence')
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(output/('training-support.'+ext),dpi=180)
    plt.close(fig)
    selected=[s for s in support['strata'] if s['direction'] in ('prefix','suffix')
              and 'update_delta_cosine_when_identical_support' in s]
    grouped=[]
    for order in sorted({s['order'] for s in selected}):
        strata=[s for s in selected if s['order']==order]
        pairs=sum(s['cosine_when_identical_support']['count'] for s in strata)
        if not pairs:continue
        final=sum(s['cosine_when_identical_support'].get('mean',0)*s['cosine_when_identical_support']['count'] for s in strata)/pairs
        delta=sum(s['update_delta_cosine_when_identical_support'].get('mean',0)*s['cosine_when_identical_support']['count'] for s in strata)/pairs
        grouped.append((order,pairs,final,delta))
    if grouped:
        fig,axes=plt.subplots(1,2,figsize=(11,4.5))
        axes[0].plot([v[0] for v in grouped],[v[2] for v in grouped],'o-',label='Final trained vectors')
        axes[0].plot([v[0] for v in grouped],[v[3] for v in grouped],'o-',label='Actual trained − initial updates')
        axes[0].set(xlabel='Source n-gram character length',ylabel='Mean cosine, identical word support')
        axes[0].legend(fontsize=9)
        axes[1].bar([v[0] for v in grouped],[v[1] for v in grouped])
        axes[1].set(xlabel='Source n-gram character length',ylabel='Conditional sampled ancestor pairs')
        for ax in axes:ax.grid(alpha=.2)
        fig.suptitle('Co-training signal and initialization: actual CPU SGNS rows; conditional, correlated samples')
        fig.tight_layout()
        for ext in ('png','svg'):fig.savefig(output/('co-training-deltas.'+ext),dpi=180)
        plt.close(fig)


def plots(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for direction in ('prefix','suffix','sibling-prefix','sibling-suffix'):
        selected=[s for s in summary['pair_strata'] if s['direction']==direction]
        orders=sorted({s['order'] for s in selected}); mean=[]; joint=[]; norm=[]
        for order in orders:
            strata=[s for s in selected if s['order']==order]
            mass=sum(s['pairs'] for s in strata)
            mean.append(sum(s['pairs']*s['cosine']['mean'] for s in strata)/mass)
            joint.append(100*sum(s['pairs']*s['fixed099']['joint_fraction'] for s in strata)/mass)
            norm.append(100*sum(s['pairs']*s['fixed099']['ratio_only_fraction'] for s in strata)/mass)
        axes[0].plot(orders,mean,'o-',label=direction)
        axes[1].plot(orders,joint,'o-',label=direction)
        axes[2].plot(orders,norm,'o-',label=direction)
    axes[0].set_ylabel('Mean cosine of sampled real pairs')
    axes[1].set_ylabel('Cosine ≥ .99 AND norm ratio ≥ .99 (%)')
    axes[2].set_ylabel('Norm ratio ≥ .99 alone (%)')
    for ax in axes:
        ax.set_xlabel('Source n-gram character length'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('One finite order-12 SGNS table; ancestor samples and nearest-sibling diagnostics differ')
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(output/('affix-audit.'+ext),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for encoding in ('fp32','fp16','int8'):
        variants=sorted([v for v in summary['variants'] if v['encoding']==encoding and v['policy']=='joint'],key=lambda v:v['threshold'])
        threshold=[v['threshold'] for v in variants]
        extra=[100*v['additional_saving_vs_best_same_encoding_control'] for v in variants]
        error=[100*v['query_weighted_relative_frobenius'] for v in variants]
        merged=[100*v['fidelity']['merged_rows']/v['fidelity']['rows'] for v in variants]
        axes[0].plot(threshold,extra,'o-',label=encoding)
        axes[1].plot(error,extra,'o-',label=encoding)
        axes[2].plot(threshold,merged,'o-',label=encoding)
    axes[0].set(xlabel='Cosine threshold (norm ratio fixed .99)',ylabel='Additional complete backend saving (%)')
    axes[1].set(xlabel='Held-out-query weighted FP32 error (%)',ylabel='Additional complete backend saving (%)')
    axes[2].set(xlabel='Cosine threshold (norm ratio fixed .99)',ylabel='Rows merged (%)')
    for ax in axes:
        ax.axhline(0,color='black',linewidth=.6);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Production DATrie + mark-compact; negative savings retained; vector error is not task quality')
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(output/('structural-tradeoffs.'+ext),dpi=180)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--plots',action='store_true')
    p.add_argument('--training-support',type=Path)
    cfg=p.parse_args()
    data=gzip.decompress(cfg.report.read_bytes()) if cfg.report.suffix=='.gz' else cfg.report.read_bytes()
    summary=analyze(json.loads(data))
    cfg.output_dir.mkdir(parents=True,exist_ok=True)
    (cfg.output_dir/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    if cfg.plots:plots(summary,cfg.output_dir)
    if cfg.training_support:
        raw=gzip.decompress(cfg.training_support.read_bytes()) if cfg.training_support.suffix=='.gz' else cfg.training_support.read_bytes()
        support=json.loads(raw)
        if support['source_sha256']!=summary['source_sha256'] or support['status']!='passed':
            raise ValueError('training support source mismatch')
        plot_training_support(support,cfg.output_dir)


if __name__=='__main__':
    main()
