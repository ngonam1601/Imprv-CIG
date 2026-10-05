"""Separate BLIP caption experiment; existing scripts and caches stay untouched."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import clip
from transformers import BlipProcessor, BlipForConditionalGeneration
from cache_cirr_query_embeddings import compose_text_prompt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'analysis/blip'
MODEL='Salesforce/blip-image-captioning-base'

def load_split(split):
    triplets=json.loads((ROOT/f'data/cirr/captions/cap.rc2.{split}.json').read_text())
    paths=json.loads((ROOT/f'data/cirr/image_splits/split.rc2.{split}.json').read_text())
    return triplets,paths

@torch.no_grad()
def generate_captions():
    processor=BlipProcessor.from_pretrained(MODEL,cache_dir=ROOT/'data/hf_cache_blip')
    model=BlipForConditionalGeneration.from_pretrained(MODEL,cache_dir=ROOT/'data/hf_cache_blip',torch_dtype=torch.float16).cuda().eval()
    for split in ['val','test1']:
        triplets,paths=load_split(split)
        dest=OUT/f'captions_ref_{split}.json'
        saved=json.loads(dest.read_text()) if dest.exists() else {'model':MODEL,'generation':{'num_beams':3,'max_new_tokens':32,'do_sample':False},'captions':{}}
        names=list(dict.fromkeys(t['reference'] for t in triplets))
        pending=[name for name in names if name not in saved['captions']]
        print(split,'unique references',len(names),'pending',len(pending),flush=True)
        for start in range(0,len(pending),32):
            batch=pending[start:start+32]
            images=[]
            for name in batch:
                with Image.open(ROOT/'data'/paths[name]) as img: images.append(img.convert('RGB'))
            inputs=processor(images=images,return_tensors='pt').to('cuda',torch.float16)
            tokens=model.generate(**inputs,num_beams=3,max_new_tokens=32,do_sample=False)
            captions=processor.batch_decode(tokens,skip_special_tokens=True)
            for name,caption in zip(batch,captions):
                assert caption.strip(),f'Empty BLIP caption {name}'
                saved['captions'][name]=caption.strip()
            temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(saved,indent=2));temp.replace(dest)
            if start%320==0:print(split,'captioned',len(saved['captions']),'/',len(names),flush=True)
    del model;torch.cuda.empty_cache()

@torch.no_grad()
def encode_captions():
    model,_=clip.load(str(ROOT/'ViT-B-32.pt'),device='cuda',jit=False)
    model=model.float().eval()
    for split,seeds in [('val',[1600,1601]),('test1',[1600])]:
        triplets,_=load_split(split)
        captions=json.loads((OUT/f'captions_ref_{split}.json').read_text())['captions']
        prompts=[compose_text_prompt(captions[t['reference']],t['caption']) for t in triplets]
        text=torch.cat([F.normalize(model.encode_text(clip.tokenize(prompts[i:i+64],truncate=True).cuda()),dim=-1).cpu() for i in range(0,len(prompts),64)])
        for seed in seeds:
            source=ROOT/'analysis'/ (f'cirr_three_branch_cache_seed{seed}.pt' if split=='val' else f'cirr_three_branch_cache_test1_seed{seed}.pt')
            cache=torch.load(source,map_location='cpu')
            assert cache['pair_ids']==[int(t['pairid']) for t in triplets]
            cache['text']=text;cache['text_prompts']=prompts
            cache['metadata']=dict(cache['metadata'],caption_model=MODEL,reference_caption_file=str(OUT/f'captions_ref_{split}.json'),source_cache=str(source))
            cache['metadata'].pop('reference_caption_sha256',None)
            torch.save(cache,OUT/f'cache_{split}_seed{seed}.pt')
        print('Encoded',split,tuple(text.shape),flush=True)
    del model;torch.cuda.empty_cache()

@torch.no_grad()
def rank_setup(cache,triplets):
    names=cache['index_names'];lookup={name:i for i,name in enumerate(names)}
    refs=torch.tensor([lookup[t['reference']] for t in triplets],device='cuda')
    branches=[F.normalize(cache[k].float().cuda(),dim=-1) for k in ['ref','gen','text']]
    index=F.normalize(cache['index'].float().cuda(),dim=-1)
    return names,lookup,refs,branches,index

def rank(w,refs,branches,index):
    q=F.normalize(sum(a*b for a,b in zip(w,branches)),dim=-1)
    scores=q@index.T
    scores[torch.arange(len(refs),device='cuda'),refs]=-torch.inf
    return torch.argsort(1-scores,dim=-1)

@torch.no_grad()
def evaluate():
    previous=json.loads((ROOT/'analysis/cirr_three_branch_weight_sweep.json').read_text())
    ratios=[r['ratio'] for r in previous['results'] if r['seed']==1600]
    triplets,_=load_split('val');results=[]
    for seed in [1600,1601]:
        cache=torch.load(OUT/f'cache_val_seed{seed}.pt',map_location='cpu')
        names,lookup,refs,branches,index=rank_setup(cache,triplets)
        targets=torch.tensor([lookup[t['target_hard']] for t in triplets],device='cuda')
        groups=torch.zeros((len(triplets),len(names)),dtype=torch.bool,device='cuda')
        for i,t in enumerate(triplets):groups[i,[lookup[n] for n in t['img_set']['members']]]=True
        groups[torch.arange(len(refs),device='cuda'),refs]=False
        for w in ratios:
            order=rank(w,refs,branches,index)
            ranks=(order==targets[:,None]).int().argmax(dim=1)
            gr=(groups.gather(1,order)&(torch.arange(len(names),device='cuda')[None,:]<ranks[:,None])).sum(dim=1)
            result={'seed':seed,'ratio':w,**{f'recall_at{k}':(ranks<k).float().mean().item()*100 for k in [1,5,10,50]},**{f'group_recall_at{k}':(gr<k).float().mean().item()*100 for k in [1,2,3]}}
            results.append(result)
        print('Evaluated',seed,len(ratios),'ratios',flush=True)
    (OUT/'validation_weight_sweep.json').write_text(json.dumps({'caption_model':MODEL,'branch_order':['ref','gen','caption'],'results':results},indent=2))
    combined=[]
    for w in ratios:
        rs=[r for r in results if r['ratio']==w]
        combined.append({'ratio':w,**{k:sum(r[k] for r in rs)/len(rs) for k in ['recall_at1','recall_at5','recall_at10','recall_at50','group_recall_at1']}})
    combined.sort(key=lambda r:r['recall_at1'],reverse=True)
    print('BLIP TOP',json.dumps(combined[:5]),flush=True)
    requested=[[0,0,1],[1,0,1],[0,1,1],[1,1,1],[.5,.5,1],[.5,.5,0],[.4,.3,.3]]
    selected=requested+[r['ratio'] for r in combined[:3] if r['ratio'] not in requested]
    lines=['# BLIP reference-caption experiment','','Model: Salesforce/blip-image-captioning-base; beam search 3, max 32 new tokens. Only caption branch changed; same ref/gen/candidate caches. Caption ref + delta encoded with CLIP ViT-B/32. Metrics below average validation seeds 1600 and 1601. Weights selected on validation.','','| Ref/gen/caption | BLIP R@1 | Old R@1 | BLIP R@10 | Old R@10 | BLIP R@50 |','|---|---:|---:|---:|---:|---:|']
    for w in selected:
        norm=[round(a/sum(w),8) for a in w]
        old=[r for r in previous['results'] if r['normalized_weights']==norm]
        new=[r for r in results if r['ratio']==w]
        if not new:
            new=[r for r in results if [round(a/sum(r['ratio']),8) for a in r['ratio']]==norm]
        mean=lambda rows,key:sum(r[key] for r in rows)/len(rows)
        lines.append('| '+ '/'.join(f'{a:g}' for a in w)+f" | {mean(new,'recall_at1'):.2f} | {mean(old,'recall_at1'):.2f} | {mean(new,'recall_at10'):.2f} | {mean(old,'recall_at10'):.2f} | {mean(new,'recall_at50'):.2f} |")
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    return selected

@torch.no_grad()
def export_test(selected):
    triplets,_=load_split('test1')
    cache=torch.load(OUT/'cache_test1_seed1600.pt',map_location='cpu')
    names,lookup,refs,branches,index=rank_setup(cache,triplets)
    names=np.asarray(names);dest=OUT/'test1_seed1600_submissions';dest.mkdir(exist_ok=True)
    manifest={'caption_model':MODEL,'branch_order':['ref','gen','caption'],'submissions':[]}
    for w in selected:
        order=rank(w,refs,branches,index).cpu().numpy()
        full={'version':'rc2','metric':'recall'};subset={'version':'rc2','metric':'recall_subset'}
        for t,row in zip(triplets,names[order]):
            key=str(t['pairid']);full[key]=row[:50].tolist()
            group=set(t['img_set']['members'])-{t['reference']}
            subset[key]=[name for name in row if name in group][:3]
            assert len(set(full[key]))==50 and t['reference'] not in full[key]
            assert len(set(subset[key]))==3
        label='_'.join(f'{a:g}'.replace('.','p') for a in w)
        name=f'cirr_test1_seed1600_blip_{label}.json'
        (dest/name).write_text(json.dumps(full));(dest/('subset_'+name)).write_text(json.dumps(subset))
        manifest['submissions'].append({'ratio':w,'full':name,'subset':'subset_'+name})
        print('Exported BLIP test1',w,flush=True)
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    generate_captions();encode_captions();export_test(evaluate())
