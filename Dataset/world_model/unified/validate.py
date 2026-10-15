"""Paired prediction-quality evaluation on real held-out windows."""
import argparse
import json
from pathlib import Path
from Dataset.world_model.unified import data as D
from Dataset.world_model.unified.value_head import load_selector
from Dataset.world_model.unified.supervise import frozen_predictor
from Dataset.world_model.unified.evaluate import score_jobs, summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selector'); p.add_argument('--predictor', required=True); p.add_argument('--jobs', required=True)
    p.add_argument('--windows', type=int, default=6); p.add_argument('--output', required=True)
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=1)
    p.add_argument('--merge', action='store_true'); args = p.parse_args()
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    if args.merge:
        rows = [json.loads(line) for i in range(args.shards)
                for line in out.with_suffix(f'.{i}.jsonl').read_text().splitlines()]
    else:
        jobs = json.loads(Path(args.jobs).read_text())[:args.windows]
        if any(j['split'] == 'train' for j in jobs):
            raise ValueError('evaluation requires DEV or TEST jobs')
        predictor = frozen_predictor(args.predictor, {'predictor_lora_rank': 16, 'predictor_query_chunk': 8})
        selector = load_selector(args.selector) if args.selector else None
        path = out.with_suffix(f'.{args.shard}.jsonl')
        with path.open('w') as log:
            def sink(row):
                log.write(json.dumps(row) + '\n'); log.flush()
                print(json.dumps({k: row[k] for k in ('job_id', 'policy', 'loss')}), flush=True)
            score_jobs(selector, predictor, D.Encoder(), jobs[args.shard::args.shards],
                       ('selector', 'recency', 'relevance') if selector else ('recency', 'relevance'), sink=sink)
        if args.shards != 1:
            return
        rows = [json.loads(line) for line in path.read_text().splitlines()]
    out.write_text(json.dumps({'selector': args.selector, 'predictor': args.predictor,
                              **summarize(rows), 'rows': rows}, indent=2) + '\n')


if __name__ == '__main__':
    main()
