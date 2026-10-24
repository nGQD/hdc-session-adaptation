"""Independent sign-enumeration, non-inferiority, and Pareto checks from saved streams."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT/'mjcs/results'
EXP = ROOT/'ieee_access/experiments/results'
BASE = EXP/'baseline_robustness'
SEEDS = [101,113,127,131,137,149,157,163,173,181]
TASKS = ['h1','m1','m2']
GRAD = ['finetune_full','finetune_readout','ewc','replay','full_tag','readout_tag','last_tag']
ANALYTIC = ['frozen','closed_linear','closed_zero','hdc','hdc_zero','hdc_shared','hdc_bound']
METHODS = ['frozen',*GRAD,*ANALYTIC[1:]]
REFS = ['finetune_full','replay','full_tag','readout_tag']
LOCAL = ['rnn_zeroshot','rnn_oracle','rnn_finetune','wf_zeroshot','wf_oracle','wf_continual']
SOURCES = {}

def read(path):
    raw=path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
    return json.loads(raw)

def holm(ps):
    order=np.argsort(ps); result=np.empty(len(ps));floor=0.
    for rank,index in enumerate(order):
        floor=max(floor,(len(ps)-rank)*ps[index]);result[index]=min(1.,floor)
    return result.tolist()

def exact_signed_rank(d):
    d=np.asarray(d);d=d[d!=0]
    if not len(d):return 1.
    ranks=stats.rankdata(abs(d));obs=abs(np.sum(np.sign(d)*ranks))
    null=np.abs(np.asarray(list(itertools.product([-1,1],repeat=len(d))))@ranks)
    return float(np.mean(null>=obs-1e-12))

def correct_resources(record, task, method, meta, selected):
    count=meta['held_in'];D=selected['hdc']['dimension'];o=len(meta['target_mean'])
    prior=meta['prior_width'];M=meta['model_parameters'];L=prior*o
    H=meta['projection_width']*256+256+256*128+128+L
    memory=[]
    for step,native in enumerate(record['memory'],1):
        added=0
        if method in ANALYTIC[1:]:added+=L
        if method=='hdc_shared':added+=(prior+2*D)*o
        if method=='hdc_bound':added+=step*D
        if method in ['finetune_full','ewc','replay']:added+=M
        if method in ['readout_tag','last_tag']:added+=H
        if method=='last_tag':added+=step*(H-L)
        if method=='replay':added+=sum(r['prefix_bins'] for r in meta['sessions'][:count+step])
        memory.append(native+added)
    costs=record['adapt_macs']
    if method in ANALYTIC[1:]:
        dim=D if method.startswith('hdc') else 0;d=129+2*dim
        source=128 if selected['hdc']['source']=='penultimate' else meta['projection_width']
        costs=[]
        for row in meta['sessions'][count:]:
            N=row['fit_bins'];q=d if method=='hdc_shared' else min(N,d)
            costs.append(row['prefix_bins']*sum(record['backbone']['macs'].values())
                         +N*source*dim+N*d*q+N*d*o+q**3//6+q*q*o)
    return float(np.mean(costs)),memory[-1],costs

def run_metrics(record):
    R=np.asarray(record['R'],float);k=record['held_in'];T=R.shape[0]-1
    assert R.shape==(T+1,k+T) and np.isfinite(R).all()
    last=R[-1];diag=R[np.arange(T)+1,k+np.arange(T)]
    m=dict(acc=last.mean(),acc_held_out=last[k:].mean(),acc_held_in=last[:k].mean(),
           bwt=((last[:k]-R[0,:k]).sum()+(last[k:k+T-1]-diag[:-1]).sum())/(k+T-1),
           plasticity=diag.mean())
    assert all(abs(float(value)-record['summary'][metric])<1e-12 for metric,value in m.items())
    return {key:float(v) for key,v in m.items()}

def main():
    report=dict(kind='independent_numerical_audit_not_production_tables',complete=False,
                seeds=SEEDS,tasks={},pending_tasks=[],noninferiority=[],multiplicity_counts=[])
    solver=read(EXP/'efficient_closed_form/summary.json')
    for task in TASKS:
        required=[BASE/'stream'/task/'full'/f'{m}_s{s}.json' for m in GRAD for s in SEEDS]
        if not all(p.exists() for p in required):
            report['pending_tasks'].append(dict(task=task,missing_selected_records=sum(not p.exists() for p in required)))
            continue
        selections=read(BASE/'selection'/f'{task}.json')
        original_sel=read(OLD/'selection'/f'{task}.json')
        assert selections['complete']
        metadata={s:read(EXP/'feature_controls/features'/task/'outer'/f's{s}'/'metadata.json') for s in SEEDS}
        data={}; rows={};raw_by_method={};cost_by_method={}
        for method in METHODS:
            values=[];raw_by_method[method]=[];cost_by_method[method]=[]
            for seed in SEEDS:
                p=(BASE/'stream' if method in GRAD else OLD/'stream')/task/'full'/f'{method}_s{seed}.json'
                r=read(p)
                assert r['task']==task and r['method']==method and r['seed']==seed
                assert r['names']==[z['name'] for z in metadata[seed]['sessions']]
                if method in GRAD:
                    candidate=selections['methods'][method]['selected']
                    assert r['candidate_id']==candidate['id'] and r['config']==candidate['config']
                v=run_metrics(r)
                v['adapt_macs'],v['memory'],costs=correct_resources(r,task,method,metadata[seed],original_sel)
                raw_by_method[method].append(r);cost_by_method[method].append(costs)
                values.append(v)
            data[method]={metric:np.asarray([v[metric] for v in values]) for metric in values[0]}
            rows[method]={metric:dict(mean=float(a.mean()),sd=float(a.std(ddof=1)),per_seed=a.tolist())
                          for metric,a in data[method].items()}
            if method in solver['tasks'][task]:
                assert abs(rows[method]['adapt_macs']['mean']-solver['tasks'][task][method]['mean_adapt_macs'])<1e-6
        paired=[]
        for metric in ['acc','bwt']:
            cells=[]
            for method in METHODS:
                if method=='hdc':continue
                d=data['hdc'][metric]-data[method][metric]
                p=exact_signed_rank(d)
                if np.any(d):assert abs(p-stats.wilcoxon(d).pvalue)<1e-14
                cells.append(dict(reference=method,metric=metric,mean_difference=float(d.mean()),
                                  minimum_seed_difference=float(d.min()),maximum_seed_difference=float(d.max()),
                                  positive_seeds=int(np.sum(d>0)),zero_seeds=int(np.sum(d==0)),p=p))
            for cell,adjusted in zip(cells,holm([c['p'] for c in cells])):
                cell['holm_13_p']=adjusted
            paired+=cells
        local={}
        for method in LOCAL:
            values=[run_metrics(read(OLD/'stream'/task/'falcon'/f'{method}_s{s}.json')) for s in SEEDS]
            local[method]={metric:float(np.mean([v[metric] for v in values])) for metric in values[0]}
            local[method]['hdc_minus_local_acc']=rows['hdc']['acc']['mean']-local[method]['acc']
        selection_changes={}
        for method in GRAD:
            sel=selections['methods'][method];inc=sel['incumbent'];values=[]
            for seed in SEEDS:
                r=read(BASE/'outer'/task/method/f"{inc['id']}_s{seed}.json")
                v=run_metrics(r);v['adapt_macs'],v['memory'],_=correct_resources(r,task,method,metadata[seed],original_sel)
                values.append(v)
            inc_means={metric:float(np.mean([v[metric] for v in values])) for metric in values[0]}
            selection_changes[method]=dict(selected_config=sel['selected']['config'],incumbent_config=inc['config'],
                                          inner_acc_gain=sel['inner_acc_gain'],changed=sel['changed'],
                                          incumbent_outer=inc_means,
                                          selected_minus_incumbent_outer_acc=rows[method]['acc']['mean']-inc_means['acc'],
                                          selected_over_hdc_macs=rows[method]['adapt_macs']['mean']/rows['hdc']['adapt_macs']['mean'],
                                          incumbent_over_hdc_macs=inc_means['adapt_macs']/rows['hdc']['adapt_macs']['mean'])
        # Independent paired crossed bootstrap, with the published engine's seed
        # and draw order solely to permit exact numerical comparison later.
        k=metadata[SEEDS[0]]['held_in'];T=len(metadata[SEEDS[0]]['sessions'])-k
        rng=np.random.default_rng(0);B=10000
        ins=rng.integers(0,k,(B,k));outs=rng.integers(0,T,(B,T))
        seed_ix=(rng.random((B,10))*10).astype(int);drawn=[]
        for method in METHODS:
            R=np.asarray([r['R'] for r in raw_by_method[method]])
            fin=R[:,-1,:k][seed_ix[:,:,None],ins[:,None,:]]
            fout=R[:,-1,k:][seed_ix[:,:,None],outs[:,None,:]]
            first=R[:,0,:k][seed_ix[:,:,None],ins[:,None,:]]
            diag=R[:,np.arange(T)+1,k+np.arange(T)]
            delta=(R[:,-1,k:]-diag)[seed_ix[:,:,None],outs[:,None,:]]
            valid=(outs<T-1)[:,None,:]
            bwt=((fin-first).sum(-1)+(delta*valid).sum(-1))/(k+valid.sum(-1))
            acc=(fin.sum(-1)+fout.sum(-1))/(k+T)
            cost=np.asarray(cost_by_method[method])[seed_ix[:,:,None],outs[:,None,:]]
            metrics=dict(acc=acc.mean(1),acc_held_out=fout.mean((1,2)),
                         bwt=bwt.mean(1),adapt_macs=cost.mean((1,2)))
            for metric,boot in metrics.items():
                rows[method][metric]['ci95_crossed_bootstrap']=np.quantile(boot,[.025,.975]).tolist()
            drawn.append(np.stack([metrics['acc'],metrics['bwt'],-metrics['adapt_macs']],axis=1))
        X=np.stack(drawn,axis=1);nondom=np.ones((B,len(METHODS)),bool)
        point=np.asarray([[rows[m]['acc']['mean'],rows[m]['bwt']['mean'],-rows[m]['adapt_macs']['mean']] for m in METHODS])
        point_nd=np.ones(len(METHODS),bool)
        for i in range(len(METHODS)):
            for j in range(len(METHODS)):
                if i==j:continue
                nondom[:,i]&=~(np.all(X[:,j,:]>=X[:,i,:],axis=1)&np.any(X[:,j,:]>X[:,i,:],axis=1))
                point_nd[i]&=not(np.all(point[j]>=point[i]) and np.any(point[j]>point[i]))
        pareto=dict(prob_non_dominated=dict(zip(METHODS,nondom.mean(0).tolist())),
                    non_dominated=[m for m,yes in zip(METHODS,point_nd) if yes])
        report['tasks'][task]=dict(methods=rows,paired_tests=paired,local_references=local,pareto=pareto,
                                  selection_changes=selection_changes,
                                  best_acc_methods=sorted(METHODS,key=lambda m:rows[m]['acc']['mean'],reverse=True),
                                  hdc_over_closed_macs=rows['hdc']['adapt_macs']['mean']/rows['closed_linear']['adapt_macs']['mean'],
                                  replay_over_hdc_memory=rows['replay']['memory']['mean']/rows['hdc']['memory']['mean'],
                                  held_in_sessions=metadata[SEEDS[0]]['held_in'])
        for method in REFS:
            for endpoint in ['acc','acc_held_out']:
                d=data['hdc'][endpoint]-data[method][endpoint]
                mean=float(d.mean());se=float(d.std(ddof=1)/np.sqrt(len(d)))
                for margin in [.01,.02,.03,.05]:
                    p=float(stats.t.sf((mean+margin)/se,len(d)-1)) if se else float(mean<=-margin)
                    report['noninferiority'].append(dict(task=task,reference=method,endpoint=endpoint,margin=margin,
                                                        mean_difference=mean,standard_error=se,p=p,
                                                        ci90_seed=[mean-stats.t.ppf(.95,9)*se,mean+stats.t.ppf(.95,9)*se],
                                                        p_bonferroni_12=min(1.,12*p),bonferroni_noninferior=12*p<.05))
    report['complete']=not report['pending_tasks']
    if report['complete']:
        for endpoint in ['acc','acc_held_out']:
            for margin in [.01,.02,.03,.05]:
                cells=[r for r in report['noninferiority'] if r['endpoint']==endpoint and r['margin']==margin]
                assert len(cells)==12
                adj=holm([r['p'] for r in cells])
                report['multiplicity_counts'].append(dict(endpoint=endpoint,margin=margin,
                                                          unadjusted=sum(r['p']<.05 for r in cells),
                                                          holm=sum(p<.05 for p in adj),
                                                          bonferroni=sum(r['bonferroni_noninferior'] for r in cells)))
    report['input_sha256']=SOURCES
    report['review_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (EXP/'final_numerical_review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['tasks','noninferiority','input_sha256']},indent=2))
    for task,record in report['tasks'].items():
        print(task,json.dumps(dict(means={m:{k:v[k]['mean'] for k in ['acc','acc_held_out','bwt','adapt_macs','memory']}
                                         for m,v in record['methods'].items()},pairings=record['paired_tests'],
                                    selections=record['selection_changes'],local_references=record['local_references']),indent=2))

if __name__=='__main__':main()
