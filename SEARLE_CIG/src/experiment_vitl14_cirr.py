"""Retrieval with SEARLE-XL (CLIP ViT-L/14) instead of SEARLE (ViT-B/32), same CIG images and Idefics2 captions.

Encodes ref/gen pseudo-token queries, candidate images and the Idefics2 "{description} {delta}" caption
branch with ViT-L/14, then evaluates weights x {plain, dual-softmax over own queries, dual-softmax over the
other split's query bank} and exports test1 submissions. Caches go to /root/vitl14_cache (/workspace is over
quota); reports and submissions go to analysis/vitl14/.
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import clip
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import collate_fn, targetpad_transform
from datasets import CIRRDataset
from encode_with_pseudo_tokens import encode_with_pseudo_tokens
from utils_feat import extract_image_features
from experiment_blip_cirr import ROOT, load_split
from dual_softmax_cirr import queries, scores

CACHE=Path('/root/vitl14_cache')
OUT=ROOT/'analysis/vitl14'
SOURCES={('val',1600):'data/generated_images_val',('val',1601):'data/generated_images_val_seed1601',
         ('test1',1600):'data/generated_images_test1_seed1600'}
WEIGHTS=[[1,1,0],[1,1,0.2],[1,1,1]]
TEMPERATURES=[0.01,0.015,0.02,0.03,0.05]

@torch.no_grad()
def build_caches():
    model,_=clip.load('ViT-L/14',device='cuda',jit=False,download_root=str(CACHE))
    model=model.float().eval()
    phi,_=torch.hub.load('miccunifi/SEARLE','searle',source='github',backbone='ViT-L/14')
    phi=phi.cuda().eval()
    preprocess=targetpad_transform(1.25,model.visual.input_resolution)
    index={}
    for split in ['val','test1']:
        features,names=extract_image_features(CIRRDataset(ROOT/'data',split,'classic',preprocess),model,batch_size=64,num_workers=4)
        index[split]=(F.normalize(features.float(),dim=-1).cpu(),names)
    for (split,seed),gen_dir in SOURCES.items():
        path=CACHE/f'cache_{split}_seed{seed}.pt'
        if path.exists():continue
        triplets,_=load_split(split)
        loader=DataLoader(CIRRDataset(ROOT/'data',split,'relative',preprocess,generated_image_dir=ROOT/gen_dir),batch_size=64,num_workers=4,collate_fn=collate_fn)
        ref,gen,pair_ids=[],[],[]
        for batch in tqdm(loader,desc=f'{split} seed {seed}'):
            tokens=clip.tokenize([f'a photo of $ that {c}' for c in batch['relative_caption']],truncate=True).cuda()
            ref.append(F.normalize(encode_with_pseudo_tokens(model,tokens,phi(model.encode_image(batch['reference_image'].cuda()))),dim=-1).cpu())
            gen.append(F.normalize(encode_with_pseudo_tokens(model,tokens,phi(model.encode_image(batch['generated_image'].cuda()))),dim=-1).cpu())
            pair_ids+=[int(p) for p in batch['pair_id']]
        assert pair_ids==[t['pairid'] for t in triplets]
        captions=json.loads((ROOT/f'analysis/idefics2/captions_edit_{split}.json').read_text())['captions']
        prompts=[f"{captions[str(t['pairid'])].strip()} {t['caption'].strip(' .')}" for t in triplets]
        text=torch.cat([F.normalize(model.encode_text(clip.tokenize(prompts[i:i+128],truncate=True).cuda()),dim=-1).cpu() for i in range(0,len(prompts),128)])
        torch.save({'metadata':{'clip':'ViT-L/14','phi':'searle-xl','preprocess':'targetpad','generated_image_dir':gen_dir,'caption':'idefics2 description + delta'},
                    'pair_ids':pair_ids,'ref':torch.cat(ref),'gen':torch.cat(gen),'text':text,'index':index[split][0],'index_names':index[split][1]},path)
        print('Saved',path,flush=True)

def load(split,seed):
    return torch.load(CACHE/f'cache_{split}_seed{seed}.pt',map_location='cpu')

def ranks(cache,triplets,weights,temperature,bank):
    s,lookup=scores(cache,triplets,weights,temperature,None,bank)
    targets=torch.tensor([lookup[t['target_hard']] for t in triplets]);rows=torch.arange(len(s))
    groups=torch.zeros_like(s,dtype=torch.bool)
    for i,t in enumerate(triplets):groups[i,[lookup[n] for n in t['img_set']['members']]]=True
    groups[rows,torch.tensor([lookup[t['reference']] for t in triplets])]=False
    better=s>s[rows,targets][:,None]
    return better.sum(1),(better&groups).sum(1)

def metrics(rank,group_rank,mask):
    return {**{f'recall_at{k}':(rank[mask]<k).float().mean().item()*100 for k in [1,5,10,50]},'group_recall_at1':(group_rank[mask]<1).float().mean().item()*100}

def export(cache,triplets,weights,temperature,bank,name):
    s,_=scores(cache,triplets,weights,temperature,None,bank)
    names=np.asarray(cache['index_names']);order=torch.argsort(s,dim=1,descending=True).numpy()
    full={'version':'rc2','metric':'recall'};subset={'version':'rc2','metric':'recall_subset'}
    for t,row in zip(triplets,names[order]):
        key=str(t['pairid']);full[key]=row[:50].tolist()
        group=set(t['img_set']['members'])-{t['reference']}
        subset[key]=[n for n in row if n in group][:3]
        assert len(set(full[key]))==50 and t['reference'] not in full[key] and len(set(subset[key]))==3
    dest=OUT/'test1_seed1600_submissions';dest.mkdir(parents=True,exist_ok=True)
    (dest/name).write_text(json.dumps(full));(dest/('subset_'+name)).write_text(json.dumps(subset))

if __name__=='__main__':
    torch.set_num_threads(8)
    CACHE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    build_caches()
    val,_=load_split('val');test,_=load_split('test1')
    half_a=torch.tensor([t['pairid']%2==0 for t in val]);everything=torch.ones_like(half_a)
    caches={seed:load('val',seed) for seed in [1600,1601]};test_cache=load('test1',1600)
    # Temperature on the even half only, with the caption-free 1/1/0 query.
    tuning={T:sum(metrics(*ranks(caches[s],val,[1,1,0],T,None),half_a)['recall_at1'] for s in caches)/2 for T in TEMPERATURES}
    T=max(tuning,key=tuning.get);print('Half-A R@1 by temperature',tuning,'-> T',T,flush=True)
    keys=['recall_at1','recall_at5','recall_at10','recall_at50','group_recall_at1']
    lines=['# SEARLE-XL (CLIP ViT-L/14) on CIRR validation','',f'Same CIG-XL images and Idefics2 "description + delta" captions as the ViT-B/32 runs. Dual-softmax T={T:g}, chosen on even pair ids (half-A R@1: '+', '.join(f'{k:g}: {v:.2f}' for k,v in tuning.items())+'). Full validation, mean of seeds 1600/1601. "own" = transductive over the split\'s own queries; "bank" = other split\'s queries (test1 for val, val seed 1600 for test1).','','| Ref/gen/caption | Scoring | R@1 | R@5 | R@10 | R@50 | Group R@1 |','|---|---|---:|---:|---:|---:|---:|']
    results=[]
    for w in WEIGHTS:
        label='_'.join(f'{a:g}'.replace('.','p') for a in w)
        val_bank=queries(test_cache,w);test_bank=queries(caches[1600],w)
        for mode,temperature,bank,test_bank_used,suffix in [('plain',None,None,None,''),('dsm own',T,None,None,f'_dsm_T{T:g}'),('dsm bank',T,val_bank,test_bank,f'_dsmbank_T{T:g}')]:
            rs=[metrics(*ranks(caches[s],val,w,temperature,bank),everything) for s in caches]
            mean={k:sum(r[k] for r in rs)/len(rs) for k in keys}
            results.append({'weights':w,'scoring':mode,'temperature':temperature,'per_seed':rs,'mean':mean})
            lines.append(f'| {"/".join(f"{a:g}" for a in w)} | {mode} | '+' | '.join(f'{mean[k]:.2f}' for k in keys)+' |')
            export(test_cache,test,w,temperature,test_bank_used,f"cirr_test1_seed1600_vitl14_idefics2_{label}{suffix.replace('.','p')}.json")
            print(lines[-1],flush=True)
    (OUT/'validation_results.json').write_text(json.dumps({'temperature':T,'tuning':tuning,'results':results},indent=2))
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print('DONE',flush=True)
