"""Sweep weighted CIRR queries using cached embeddings; no image generation."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

@torch.no_grad()
def main():
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[2]
    out = root / 'analysis'
    triplets = json.loads((root / 'data/cirr/captions/cap.rc2.val.json').read_text())
    requested = [(0,0,1),(1,0,1),(0,1,1),(1,1,1),(.5,.5,1)]
    ratios = {tuple(round(x/sum(w),8) for x in w): list(w) for w in requested}
    # Simplex grid at 0.1 resolution, plus the explicitly requested ratios.
    for a in range(11):
        for b in range(11-a):
            w=(a/10,b/10,(10-a-b)/10)
            ratios.setdefault(tuple(round(x,8) for x in w),list(w))
    results=[]
    for seed in [1600,1601]:
        cache=torch.load(out/f'cirr_three_branch_cache_seed{seed}.pt',map_location='cpu')
        assert cache['pair_ids']==[int(t['pairid']) for t in triplets]
        names=cache['index_names']; lookup={n:i for i,n in enumerate(names)}
        refs=torch.tensor([lookup[t['reference']] for t in triplets],device='cuda')
        targets=torch.tensor([lookup[t['target_hard']] for t in triplets],device='cuda')
        groups=torch.zeros((len(triplets),len(names)),dtype=torch.bool,device='cuda')
        for i,t in enumerate(triplets):
            groups[i,[lookup[n] for n in t['img_set']['members']]]=True
        rows=torch.arange(len(triplets),device='cuda')
        groups[rows,refs]=False
        index=F.normalize(cache['index'].float().cuda(),dim=-1)
        branches=[F.normalize(cache[k].float().cuda(),dim=-1) for k in ['ref','gen','text']]
        for j,(norm,w) in enumerate(ratios.items()):
            q=F.normalize(sum(x*v for x,v in zip(norm,branches)),dim=-1)
            scores=q@index.T
            scores[rows,refs]=-torch.inf
            target_scores=scores[rows,targets,None]
            # Use the same argsort ranking as the existing evaluator, including ties.
            ties=((scores==target_scores).sum(dim=1)>1).sum().item()
            order=torch.argsort(1 - scores, dim=-1)
            ranks=(order==targets[:,None]).int().argmax(dim=1)
            group_ranks=(groups.gather(1,order) & (torch.arange(len(names),device='cuda')[None,:]<ranks[:,None])).sum(dim=1)
            metrics={f'recall_at{k}':(ranks<k).float().mean().item()*100 for k in [1,5,10,50]}
            metrics.update({f'group_recall_at{k}':(group_ranks<k).float().mean().item()*100 for k in [1,2,3]})
            results.append({'seed':seed,'ratio':w,'normalized_weights':list(norm),'target_score_ties':ties,**metrics})
            if j%15==0:print(seed,j+1,'/',len(ratios),flush=True)
    payload={'branch_order':['ref','gen','caption'],'query_count':len(triplets),'requested_ratios':requested,'grid_step':.1,'results':results}
    (out/'cirr_three_branch_weight_sweep.json').write_text(json.dumps(payload,indent=2))
    combined=[]
    for norm,w in ratios.items():
        rs=[r for r in results if r['normalized_weights']==list(norm)]
        combined.append({'ratio':w,**{k:sum(r[k] for r in rs)/len(rs) for k in ['recall_at1','recall_at5','recall_at10','recall_at50','group_recall_at1']}})
    combined.sort(key=lambda r:r['recall_at1'],reverse=True)
    lines=['# CIRR three-branch weight sweep','', 'Order: ref / gen / caption. Normalize each branch, weighted sum, normalize query. 4,181 validation queries; seeds 1600 and 1601. Grid step 0.1. Results selected on validation, not independent test.','', '## Requested ratios','', '| Ratio | Seed | R@1 | R@5 | R@10 | R@50 | Group R@1 |','|---|---|---:|---:|---:|---:|---:|']
    for w in requested:
        norm=[round(x/sum(w),8) for x in w]
        for r in results:
            if r['normalized_weights']==norm:
                lines.append('| '+ '/'.join(str(x) for x in w)+f" | {r['seed']} | "+' | '.join(f'{r[k]:.2f}' for k in ['recall_at1','recall_at5','recall_at10','recall_at50','group_recall_at1'])+' |')
    lines+=['','## Top 10 by mean R@1 across seeds','','| Ratio | R@1 | R@5 | R@10 | R@50 | Group R@1 |','|---|---:|---:|---:|---:|---:|']
    for r in combined[:10]:
        lines.append('| '+ '/'.join(str(x) for x in r['ratio'])+' | '+' | '.join(f'{r[k]:.2f}' for k in ['recall_at1','recall_at5','recall_at10','recall_at50','group_recall_at1'])+' |')
    (out/'cirr_three_branch_weight_sweep.md').write_text('\n'.join(lines)+'\n')
    print('TOP',json.dumps(combined[:5]),flush=True)
    print('Target ties:',sum(r['target_score_ties'] for r in results),flush=True)

if __name__=='__main__':main()
