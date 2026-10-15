"""Continue the unified 2B predictor from a preserved checkpoint on explicit TRAIN jobs."""
import argparse
import datetime
import json
import os
from pathlib import Path
import random
import time
import torch
from Dataset.world_model.unified import assemble as A, data as D, arms, windows as W
from Dataset.world_model.unified.value_head import load_selector
from Dataset.world_model.unified.supervise import scoring_window
from Dataset.world_model.unified.train_predictor import item_losses,load_trainable,trainable_state
from Dataset.world_model.unified.predictor import Predictor
from Dataset.world_model.unified.distributed import all_sum,reduce_gradients,gather
from Dataset.world_model.unified.io import write_json


def item_from(sc,kept):
    if sc.pair is None:
        return sc.enc.item(sc.episode,sc.origin,kept=kept,query_ids=sc.qids,budget=sc.total,
                           prepared=(sc.w,sc.pw),features=sc.features)
    return arms.arm_item(sc.enc,sc.pair,sc.which,sc.total,'relevance',kept=kept)


def predictor_fit(args, rank, world):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    jobs=json.loads(Path(args.jobs).read_text())[:args.windows]
    assert all(j['split']=='train' and D.split_of(j['episode'])=='train' for j in jobs)
    torch.manual_seed(args.seed)
    model=Predictor(query_chunk=4);load_trainable(model,args.init)
    params=model.trainable_parameters();opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=.01)
    enc=D.Encoder();step=0;epoch0=cursor0=0
    identity={'args':vars(args),'world':world}
    state_path=out/'state.pt'
    if state_path.exists():
        saved=torch.load(state_path,map_location='cpu',weights_only=False)
        if saved['identity']!=identity:raise ValueError('predictor resume inputs differ')
        model.load_state_dict(saved['weights'],strict=False);opt.load_state_dict(saved['optimizer'])
        epoch0,cursor0,step=saved['epoch'],saved['cursor'],saved['step']
        torch.set_rng_state(saved['rngs'][rank][0]);torch.cuda.set_rng_state(saved['rngs'][rank][1])
    def save(epoch,cursor,name):
        rngs=gather((torch.get_rng_state(),torch.cuda.get_rng_state()),world)
        if rank==0:
            weights=trainable_state(model)
            target=out/name;tmp=target.with_suffix('.tmp');torch.save(weights,tmp);tmp.replace(target)
            tmp=state_path.with_suffix('.tmp')
            torch.save({'identity':identity,'weights':weights,'optimizer':opt.state_dict(),'epoch':epoch,
                        'cursor':cursor,'step':step,'rngs':rngs},tmp);tmp.replace(state_path)
            write_json(out/'config.json',identity)
    selector=load_selector(args.selector) if args.selector else None
    if selector:selector.eval()
    log=(out/f'train.rank{rank}.jsonl').open('a')
    for epoch in range(epoch0,args.epochs):
        order=list(jobs);random.Random(args.seed+epoch).shuffle(order)
        for cursor in range(cursor0 if epoch==epoch0 else 0,len(order),world):
            job=order[cursor+rank] if cursor+rank<len(order) else None
            opt.zero_grad(set_to_none=True);counts=torch.zeros(2,dtype=torch.float64)
            if job:
                sc=scoring_window(None,enc,job)
                kept=sorted(A.select(sc.pw,sc.qids,sc.budget,'relevance')[0])
                if selector and (cursor//world+epoch)%2==0:
                    kept,_=selector.select(sc.pw,enc.tok,sc.qids,sc.budget,sc.actions_ids)
                if args.local_augmentation and (cursor+rank+epoch)%3:
                    # Small real-evidence perturbations, not replacement of the architecture.
                    rng=random.Random(W.stable_hash('predictor-context',args.seed,epoch,job['id']))
                    captures=(cursor+rank+epoch)%3==2
                    old=[k for k in kept if (sc.pw.units[k].kind=='capture')==captures]
                    new=[u.key for u in sc.pw.candidates(sc.qids) if u.key not in set(kept)
                         and (u.kind=='capture')==captures and (captures or u.tick<sc.origin)]
                    rng.shuffle(old);rng.shuffle(new)
                    for drop in old[:(2 if captures else 4)]:
                        for add in new:
                            proposed=sorted((set(kept)-{drop})|{add})
                            if sc.cost(proposed)<=sc.budget:
                                kept=proposed;new.remove(add);break
                if args.padding_augmentation:
                    # Real evidence and labels unchanged; avoid learning from accidental blank-line phase.
                    model.PAD_TO=(1,8,64)[(cursor+rank+epoch)%3]
                item=item_from(sc,kept);model.train();start=time.monotonic()
                loss,groups,_=item_losses(model,item,1,train=True)
                counts=torch.tensor([1,loss],dtype=torch.float64)
            counts=all_sum(counts,world);reduce_gradients(params,int(counts[0]),world)
            norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step();step+=1
            if job:
                row={'step':step,'epoch':epoch,'job':job,'loss':loss,'groups':groups,'counts':item['counts'],
                     'grad_norm':float(norm),'seconds':time.monotonic()-start}
                log.write(json.dumps(row)+'\n');log.flush()
            if rank==0:
                print(json.dumps({'step':step,'loss':float(counts[1]/counts[0])}),flush=True)
            if step%args.save_every==0:
                save(epoch,min(cursor+world,len(order)),f'step_{step:05d}.pt')
        save(epoch+1,0,f'epoch_{epoch:02d}.pt')
        if rank==0:
            torch.save(trainable_state(model),out/'predictor.pt')
    log.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True);p.add_argument('--init',required=True);p.add_argument('--jobs',required=True)
    p.add_argument('--selector');p.add_argument('--windows',type=int,default=162);p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--save-every',type=int,default=15)
    p.add_argument('--padding-augmentation',action='store_true');p.add_argument('--local-augmentation',action='store_true')
    args=p.parse_args();rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    if world>1:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        torch.distributed.init_process_group('gloo',timeout=datetime.timedelta(hours=2))
    torch.cuda.set_per_process_memory_fraction(.9)
    predictor_fit(args,rank,world)
    if world>1:torch.distributed.destroy_process_group()


if __name__=='__main__':main()
