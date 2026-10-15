"""Label-free full candidate selection and future probability distributions.

Flat targets have exact probabilities. Numeric vectors are autoregressive: exact
component tables are conditioned on earlier model-modal component codes, and
ancestral joint interval samples additionally retain dependence between axes.
No teacher-forced future answer is used in prediction.
"""
import argparse
import json
import time

import torch

from Dataset.world_model.unified import data as D, codes as K, schema as S, visual as V, assemble as A, arms
from Dataset.world_model.unified.value_head import load_selector
from Dataset.world_model.unified.supervise import frozen_predictor
from Dataset.world_model.unified.io import write_json


def tree_of(domain):
    return domain.tree if isinstance(domain,K.IntervalCode) else domain


@torch.no_grad()
def probabilities(model,ctx,queries,enc,samples=8):
    states,_=model.context_states(ctx)
    text=[enc.tok('\n'+q['text'],add_special_tokens=False)['input_ids'] for q in queries]
    domains=[enc.answers.domains(q) for q in queries]
    prefixes=[list(ids) for ids in text]
    result=[{'query':q['text'],'group':q['group'],'entity':q.get('entity'),'tuple':q.get('tuple'),
             'horizon_seconds':q.get('horizon',0)*S.TICK_S,'components':[]} for q in queries]
    def forward(tasks):
        lps=[]
        for start in range(0,len(tasks),model.query_chunk):
            lps.extend(model.continuation(ctx,states,tasks[start:start+model.query_chunk]))
        return lps
    for component in range(max(map(len,domains))):
        tasks,owners=[],[]
        for i,ds in enumerate(domains):
            if component>=len(ds):continue
            tree=tree_of(ds[component])
            for prefix in tree.children:
                ids=prefixes[i]+[t for t,_ in enc.vocab.code_tokens(tree,prefix)]
                tasks.append((ids,[(len(ids)-1,enc.vocab.tokens(tree,prefix))]));owners.append((i,prefix))
        node={owner:lp[0].float().cpu() for owner,lp in zip(owners,forward(tasks))}
        for i,ds in enumerate(domains):
            if component>=len(ds):continue
            domain=ds[component];tree=tree_of(domain)
            probs=[]
            for code in tree.leaves:
                lp=sum(node[i,code[:j]][tree.children[code[:j]].index(c)] for j,c in enumerate(code))
                probs.append(float(lp.exp()))
            mass=sum(probs)
            if abs(mass-1.)>1e-4:raise RuntimeError(f'probability mass {mass} for {queries[i]["text"]}')
            entry={'codes':tree.leaves,'probabilities':probs,'mass':mass}
            if isinstance(domain,K.IntervalCode):
                bounds=domain.real_bounds();ref=queries[i]['reference'][component]
                entry.update(reference=ref,residual_intervals=bounds,
                             value_intervals=[[ref+lo,ref+hi] for lo,hi in bounds],
                             conditioning='previous model-modal component codes, not future truth')
            mode=tree.leaves[max(range(len(probs)),key=probs.__getitem__)]
            entry['modal_code']=mode;result[i]['components'].append(entry)
            prefixes[i]+=[t for t,_ in enc.vocab.code_tokens(tree,mode)]
    if samples:
        generator=torch.Generator().manual_seed(0)
        draws=[]
        for i,ds in enumerate(domains):
            if isinstance(ds[0],K.IntervalCode):
                for sample in range(samples):draws.append({'i':i,'sample':sample,'ids':list(text[i]),'c':0,'prefix':(), 'codes':[]})
        active=list(draws)
        while active:
            tasks=[]
            for d in active:
                tree=tree_of(domains[d['i']][d['c']]);ids=d['ids']
                tasks.append((ids,[(len(ids)-1,enc.vocab.tokens(tree,d['prefix']))]))
            next_active=[]
            for d,lp in zip(active,forward(tasks)):
                tree=tree_of(domains[d['i']][d['c']]);prefix=d['prefix']
                j=int(torch.multinomial(lp[0].float().cpu().exp(),1,generator=generator))
                d['ids'].append(enc.vocab.tokens(tree,prefix)[j]);d['prefix']=prefix+(tree.children[prefix][j],)
                if d['prefix'] not in tree.children:
                    d['codes'].append(d['prefix']);d['prefix']=();d['c']+=1
                if d['c']<len(domains[d['i']]):next_active.append(d)
            active=next_active
        for d in draws:
            i=d['i'];intervals=[]
            for c,code in enumerate(d['codes']):
                domain=domains[i][c];lo,hi=domain.real_bounds()[domain.codes.index(code)]
                ref=queries[i]['reference'][c];intervals.append([ref+lo,ref+hi])
            result[i].setdefault('joint_interval_samples',[]).append(intervals)
    return result


@torch.no_grad()
def predict(episode,origin,selector,predictor,enc,samples=8,arm_key=None,which='arm'):
    started=time.monotonic()
    if arm_key:
        if arms._meta(arm_key)['parent_episode']!=episode:raise ValueError('ARM belongs to another parent')
        pair={'arm_key':arm_key,'factual_key':arms._factual_key(arm_key),'parent_episode':episode,
              'group':arm_key.rsplit('/',1)[0],'origin':origin}
        w,pw,_=arms.arm_window(enc,pair);qids=list(range(len(w['queries'])))
        budget=arms.optional_budget(enc,pair,A.TOTAL_BUDGET)
        actions=w['arm_action_ids'][arm_key if which=='arm' else pair['factual_key']]
    else:
        w,pw=enc.window(episode,origin);qids=pw.select_queries(None)
        budget=enc.optional_budget(w,pw,qids,A.TOTAL_BUDGET);actions=pw.action_ids
    # Remove every future-label field before either learned model is used.
    for q in w['queries']:
        for field in ('residual','label','compatible','none'):q.pop(field,None)
    enc.set_relevance(w,pw,qids)
    kept,info=selector.select(pw,enc.tok,qids,budget,actions_ids=actions)
    segments,counts=pw.assemble(kept,qids,actions_ids=actions)
    features={V.ref_key(ref):V.load_features(ref) for kind,ref in segments if kind=='image'}
    ctx=predictor.build_context(segments,lambda ref:features[V.ref_key(ref)])
    reserve=D.query_reserve(enc,w['queries'],qids)
    if ctx.length+reserve>A.TOTAL_BUDGET:raise RuntimeError('prediction exceeded 20K total positions')
    result=probabilities(predictor,ctx,[w['queries'][i] for i in qids],enc,samples)
    return {'episode':episode,'origin_tick':origin,'history_seconds':5,'budget':A.TOTAL_BUDGET,
            'arm_key':arm_key,'which':which if arm_key else None,
            'entities':{a:e for e,a in w['aliases'].items()},
            'relations':{t:w['headings']['rel|'+t] for t in pw.queried_tuples(qids)},
            'candidates':len(pw.candidates(qids)),'kept':kept,'captures':counts['N_captures'],
            'context_positions':ctx.length,'query_reserve':reserve,'selector_positions':info['positions'],
            'seconds':time.monotonic()-started,'distributions':result}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selector',required=True);p.add_argument('--predictor',required=True)
    p.add_argument('--episode',required=True);p.add_argument('--origin',type=int,required=True,help='origin in 0.1s ticks')
    p.add_argument('--arm-key');p.add_argument('--which',choices=('arm','factual'),default='arm')
    p.add_argument('--samples',type=int,default=8);p.add_argument('--output',required=True)
    args=p.parse_args()
    if args.samples < 0:
        p.error('samples must be nonnegative')
    selector=load_selector(args.selector);selector.eval()
    predictor=frozen_predictor(args.predictor,{'predictor_lora_rank':16,'predictor_query_chunk':8})
    result=predict(args.episode,args.origin,selector,predictor,D.Encoder(),args.samples,args.arm_key,args.which)
    result.update(selector=args.selector,predictor=args.predictor)
    write_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('distributions','kept')}),flush=True)


if __name__=='__main__':main()
