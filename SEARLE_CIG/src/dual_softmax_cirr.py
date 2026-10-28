"""Weighted ref/gen/caption scoring of cached CIRR queries, with optional dual-softmax re-scoring.

Dual-softmax (hubness reduction): s'(q, c) = s(q, c) * softmax over all queries of s(., c) / T, rescaled by
the query count. It uses the unlabelled query bank of the same split (transductive, no ground truth).
Ref/gen/index branches come from analysis/blip caches. With --caption-dir the caption branch is
"{description} {delta}" from <caption-dir>/captions_edit_<split>.json, encoded with CLIP ViT-B/32.
Writes only under analysis/dual_softmax/.
Usage: dual_softmax_cirr.py [--weights 1 1 0] [--caption-dir analysis/idefics2] [--temperature 0.02]
"""
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import clip
from experiment_blip_cirr import ROOT, load_split

OUT=ROOT/'analysis/dual_softmax'
VAL_CACHES={1600:'analysis/blip/cache_val_seed1600.pt',1601:'analysis/blip/cache_val_seed1601.pt'}

@torch.no_grad()
def encode_captions(caption_dir,split,triplets,compose='plain',model=None):
    path=ROOT/caption_dir/f'captions_edit_{split}.json'
    if not path.exists():path=ROOT/caption_dir/f'captions_query_{split}.json'  # per-query reference captions
    captions=json.loads(path.read_text())['captions']
    if compose=='src4':  # dangkhoa-nguyen src4: "{caption}, {delta}", lowercased
        prompts=[f"{captions[str(t['pairid'])].strip().lower()}, {t['caption'].strip().lower()}" for t in triplets]
    elif compose=='only':  # dangkhoa-nguyen src5 Qwen: the target description alone
        prompts=[captions[str(t['pairid'])].strip() for t in triplets]
    else:
        prompts=[f"{captions[str(t['pairid'])].strip()} {t['caption'].strip(' .')}" for t in triplets]
    if model is None:
        model,_=clip.load(str(ROOT/'ViT-B-32.pt'),device='cpu',jit=False)
        model=model.float().eval()
    device=next(model.parameters()).device
    return torch.cat([F.normalize(model.encode_text(clip.tokenize(prompts[i:i+256],truncate=True).to(device)),dim=-1).cpu() for i in range(0,len(prompts),256)])

def queries(cache,weights,text=None):
    if text is not None:cache=dict(cache,text=text)
    branches=[F.normalize(cache[k].float(),dim=-1) for k in ['ref','gen','text']]
    return F.normalize(sum(w*b for w,b in zip(weights,branches)),dim=-1)

def scores(cache,triplets,weights,temperature,text=None,bank=None):
    """bank=None: transductive softmax over this split's queries. bank=tensor: each query is normalised
    independently against a fixed bank of other queries plus itself (QB-Norm style)."""
    lookup={n:i for i,n in enumerate(cache['index_names'])}
    refs=torch.tensor([lookup[t['reference']] for t in triplets])
    index=F.normalize(cache['index'].float(),dim=-1)
    s=queries(cache,weights,text)@index.T
    if temperature and bank is None:
        s=s*torch.softmax(s/temperature,dim=0)*len(s)
    elif temperature:
        z=torch.exp(bank@index.T/temperature).sum(0)
        e=torch.exp(s/temperature)
        s=s*e/(z+e)*len(bank)
    s[torch.arange(len(s)),refs]=-float('inf')
    return s,lookup

def evaluate(weights,temperature,text,bank=None):
    triplets,_=load_split('val');results=[]
    for seed,path in VAL_CACHES.items():
        cache=torch.load(ROOT/path,map_location='cpu')
        s,lookup=scores(cache,triplets,weights,temperature,text,bank)
        targets=torch.tensor([lookup[t['target_hard']] for t in triplets])
        rows=torch.arange(len(s))
        ranks=(s>s[rows,targets][:,None]).sum(1)
        groups=torch.zeros_like(s,dtype=torch.bool)
        for i,t in enumerate(triplets):groups[i,[lookup[n] for n in t['img_set']['members']]]=True
        groups[rows,torch.tensor([lookup[t['reference']] for t in triplets])]=False
        group_ranks=((s>s[rows,targets][:,None])&groups).sum(1)
        results.append({'seed':seed,'weights':weights,'temperature':temperature,'bank':'self' if bank is None else 'cross',**{f'recall_at{k}':(ranks<k).float().mean().item()*100 for k in [1,5,10,50]},**{f'group_recall_at{k}':(group_ranks<k).float().mean().item()*100 for k in [1,2,3]}})
    return results

def export_test(weights,temperature,text,tag,bank=None):
    triplets,_=load_split('test1')
    cache=torch.load(ROOT/'analysis/blip/cache_test1_seed1600.pt',map_location='cpu')
    s,_=scores(cache,triplets,weights,temperature,text,bank)
    names=np.asarray(cache['index_names']);order=torch.argsort(s,dim=1,descending=True).numpy()
    full={'version':'rc2','metric':'recall'};subset={'version':'rc2','metric':'recall_subset'}
    for t,row in zip(triplets,names[order]):
        key=str(t['pairid']);full[key]=row[:50].tolist()
        group=set(t['img_set']['members'])-{t['reference']}
        subset[key]=[name for name in row if name in group][:3]
        assert len(set(full[key]))==50 and t['reference'] not in full[key]
        assert len(set(subset[key]))==3
    dest=OUT/'test1_seed1600_submissions';dest.mkdir(parents=True,exist_ok=True)
    label='_'.join(f'{a:g}'.replace('.','p') for a in weights)
    suffix=f"_dsm{'' if bank is None else 'bank'}_T{f'{temperature:g}'.replace('.','p')}" if temperature else ''
    name=f'cirr_test1_seed1600{tag}_{label}{suffix}.json'
    (dest/name).write_text(json.dumps(full));(dest/('subset_'+name)).write_text(json.dumps(subset))
    print('Exported',dest/name,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--weights',type=float,nargs=3,default=[1,1,0],help='ref gen caption')
    parser.add_argument('--caption-dir',help='Directory with captions_edit_{val,test1}.json, e.g. analysis/idefics2')
    parser.add_argument('--temperature',type=float,default=0.02,help='Chosen on half of the validation queries')
    parser.add_argument('--compose',choices=['plain','src4','only'],default='plain',help='plain: "{caption} {delta}"; src4: "{caption}, {delta}" lowercased; only: caption alone')
    parser.add_argument('--bank',choices=['self','cross'],default='self',help="self: transductive over the split's own queries; cross: fixed bank of the other split's queries (val bank for test1, test1 bank for val)")
    args=parser.parse_args()
    torch.set_num_threads(8)
    OUT.mkdir(parents=True,exist_ok=True)
    weights=[w if w%1 else int(w) for w in args.weights]
    val_text=test_text=None;tag=''
    if args.caption_dir:
        val_text=encode_captions(args.caption_dir,'val',load_split('val')[0],args.compose)
        test_text=encode_captions(args.caption_dir,'test1',load_split('test1')[0],args.compose)
        tag='_'+args.caption_dir.rstrip('/').split('/')[-1]+('' if args.compose=='plain' else '_'+args.compose)
    elif weights[2]:
        parser.error('--caption-dir is required when the caption weight is non-zero')
    val_bank=test_bank=None
    if args.bank=='cross':
        # Banks are other-split queries only: test1 queries (unlabelled) for val, val seed 1600 for test1.
        val_bank=queries(torch.load(ROOT/'analysis/blip/cache_test1_seed1600.pt',map_location='cpu'),weights,test_text)
        test_bank=queries(torch.load(ROOT/VAL_CACHES[1600],map_location='cpu'),weights,val_text)
        tag+='_bank'
    results=evaluate(weights,None,val_text)+evaluate(weights,args.temperature,val_text,val_bank)
    report=OUT/f'report{tag}_{"_".join(f"{a:g}" for a in weights)}.md'
    (report.with_suffix('.json')).write_text(json.dumps({'caption_dir':args.caption_dir,'weights':weights,'results':results},indent=2))
    keys=['recall_at1','recall_at5','recall_at10','recall_at50','group_recall_at1']
    lines=['# Dual-softmax re-scoring on CIRR validation','',f'Ref/gen/caption {weights}; caption {args.caption_dir or "none"}. Temperature chosen on even pair ids; full validation shown. Dual-softmax is transductive: it uses the unlabelled query bank.','','| Method | Seed | R@1 | R@5 | R@10 | R@50 | Group R@1 |','|---|---|---:|---:|---:|---:|---:|']
    for temperature in [None,args.temperature]:
        rs=[r for r in results if r['temperature']==temperature]
        label='plain' if temperature is None else f'dual-softmax ({args.bank} bank) T={temperature:g}'
        for r in rs:lines.append(f"| {label} | {r['seed']} | "+' | '.join(f'{r[k]:.2f}' for k in keys)+' |')
        lines.append(f'| {label} | mean | '+' | '.join(f'{sum(r[k] for r in rs)/len(rs):.2f}' for k in keys)+' |')
    report.write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)
    if args.bank=='self':export_test(weights,None,test_text,tag)
    export_test(weights,args.temperature,test_text,tag.removesuffix('_bank'),test_bank)
