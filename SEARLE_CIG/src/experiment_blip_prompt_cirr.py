"""BLIP conditional (text-prompted) reference captions; writes only under analysis/blip_prompt/."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import clip
from transformers import BlipProcessor, BlipForConditionalGeneration
from cache_cirr_query_embeddings import compose_text_prompt
from experiment_blip_cirr import ROOT, MODEL, load_split, rank_setup, rank

OUT=ROOT/'analysis/blip_prompt'
PREFIXES=['a photography of','a picture of','an image of']

def slug(prefix):return prefix.replace(' ','_')

def strip_prefix(caption,prefix):
    caption=caption.strip()
    return caption[len(prefix):].strip() if caption.lower().startswith(prefix) else caption

@torch.no_grad()
def generate_captions():
    processor=BlipProcessor.from_pretrained(MODEL,cache_dir=ROOT/'data/hf_cache_blip')
    model=BlipForConditionalGeneration.from_pretrained(MODEL,cache_dir=ROOT/'data/hf_cache_blip',torch_dtype=torch.float16).cuda().eval()
    for prefix in PREFIXES:
        for split in ['val','test1']:
            triplets,paths=load_split(split)
            dest=OUT/slug(prefix)/f'captions_ref_{split}.json'
            dest.parent.mkdir(parents=True,exist_ok=True)
            saved=json.loads(dest.read_text()) if dest.exists() else {'model':MODEL,'prompt':prefix,'generation':{'num_beams':3,'max_new_tokens':32,'do_sample':False},'raw':{},'captions':{}}
            names=list(dict.fromkeys(t['reference'] for t in triplets))
            pending=[name for name in names if name not in saved['captions']]
            print(prefix,split,'unique references',len(names),'pending',len(pending),flush=True)
            for start in range(0,len(pending),32):
                batch=pending[start:start+32]
                images=[]
                for name in batch:
                    with Image.open(ROOT/'data'/paths[name]) as img:images.append(img.convert('RGB'))
                inputs=processor(images=images,text=[prefix]*len(batch),return_tensors='pt').to('cuda',torch.float16)
                tokens=model.generate(**inputs,num_beams=3,max_new_tokens=32,do_sample=False)
                for name,raw in zip(batch,processor.batch_decode(tokens,skip_special_tokens=True)):
                    caption=strip_prefix(raw,prefix)
                    assert caption,f'Empty BLIP caption {name}'
                    saved['raw'][name]=raw.strip();saved['captions'][name]=caption
                temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(saved,indent=2));temp.replace(dest)
    del model;torch.cuda.empty_cache()

@torch.no_grad()
def encode_captions():
    """Return {prefix: {split: text embeddings}}; ref/gen/index branches come from analysis/blip caches."""
    model,_=clip.load(str(ROOT/'ViT-B-32.pt'),device='cuda',jit=False)
    model=model.float().eval();texts={}
    for prefix in PREFIXES:
        texts[prefix]={}
        for split in ['val','test1']:
            triplets,_=load_split(split)
            captions=json.loads((OUT/slug(prefix)/f'captions_ref_{split}.json').read_text())['captions']
            prompts=[compose_text_prompt(captions[t['reference']],t['caption']) for t in triplets]
            texts[prefix][split]=torch.cat([F.normalize(model.encode_text(clip.tokenize(prompts[i:i+64],truncate=True).cuda()),dim=-1).cpu() for i in range(0,len(prompts),64)])
    del model;torch.cuda.empty_cache()
    return texts

def evaluate(texts):
    ratios=[[1,1,1]]
    triplets,_=load_split('val');results=[]
    for seed in [1600,1601]:
        cache=torch.load(ROOT/f'analysis/blip/cache_val_seed{seed}.pt',map_location='cpu')
        for prefix in PREFIXES:
            cache['text']=texts[prefix]['val']
            names,lookup,refs,branches,index=rank_setup(cache,triplets)
            targets=torch.tensor([lookup[t['target_hard']] for t in triplets],device='cuda')
            groups=torch.zeros((len(triplets),len(names)),dtype=torch.bool,device='cuda')
            for i,t in enumerate(triplets):groups[i,[lookup[n] for n in t['img_set']['members']]]=True
            groups[torch.arange(len(refs),device='cuda'),refs]=False
            for w in ratios:
                order=rank(w,refs,branches,index)
                ranks=(order==targets[:,None]).int().argmax(dim=1)
                gr=(groups.gather(1,order)&(torch.arange(len(names),device='cuda')[None,:]<ranks[:,None])).sum(dim=1)
                results.append({'prompt':prefix,'seed':seed,'ratio':w,**{f'recall_at{k}':(ranks<k).float().mean().item()*100 for k in [1,5,10,50]},**{f'group_recall_at{k}':(gr<k).float().mean().item()*100 for k in [1,2,3]}})
            print('Evaluated',seed,prefix,flush=True)
    (OUT/'validation_weight_sweep.json').write_text(json.dumps({'caption_model':MODEL,'prompts':PREFIXES,'branch_order':['ref','gen','caption'],'results':results},indent=2))
    keys=['recall_at1','recall_at5','recall_at10','recall_at50','group_recall_at1']
    combined=[]
    for prefix in PREFIXES:
        for w in ratios:
            rs=[r for r in results if r['prompt']==prefix and r['ratio']==w]
            combined.append({'prompt':prefix,'ratio':w,**{k:sum(r[k] for r in rs)/len(rs) for k in keys}})
    return combined

@torch.no_grad()
def export_test(texts,prefix,weights):
    triplets,_=load_split('test1')
    cache=torch.load(ROOT/'analysis/blip/cache_test1_seed1600.pt',map_location='cpu')
    cache['text']=texts[prefix]['test1']
    names,lookup,refs,branches,index=rank_setup(cache,triplets)
    names=np.asarray(names);dest=OUT/'test1_seed1600_submissions';dest.mkdir(exist_ok=True)
    manifest={'caption_model':MODEL,'prompt':prefix,'branch_order':['ref','gen','caption'],'submissions':[]}
    for w in weights:
        order=rank(w,refs,branches,index).cpu().numpy()
        full={'version':'rc2','metric':'recall'};subset={'version':'rc2','metric':'recall_subset'}
        for t,row in zip(triplets,names[order]):
            key=str(t['pairid']);full[key]=row[:50].tolist()
            group=set(t['img_set']['members'])-{t['reference']}
            subset[key]=[name for name in row if name in group][:3]
            assert len(set(full[key]))==50 and t['reference'] not in full[key]
            assert len(set(subset[key]))==3
        label='_'.join(f'{a:g}'.replace('.','p') for a in w)
        name=f'cirr_test1_seed1600_blipprompt_{slug(prefix)}_{label}.json'
        (dest/name).write_text(json.dumps(full));(dest/('subset_'+name)).write_text(json.dumps(subset))
        manifest['submissions'].append({'ratio':w,'full':name,'subset':'subset_'+name})
        print('Exported',prefix,w,flush=True)
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    generate_captions();texts=encode_captions();combined=evaluate(texts)
    old=json.loads((ROOT/'analysis/blip/validation_weight_sweep.json').read_text())['results']
    lines=['# BLIP prompted reference captions','',f'Model {MODEL}, conditional prefix, beam 3, max 32 new tokens; prefix stripped before "a photo of {{caption}}. {{delta}}". Mean of validation seeds 1600/1601. Unprompted = analysis/blip.','','| Prompt | Ratio | R@1 | R@5 | R@10 | R@50 | Group R@1 |','|---|---|---:|---:|---:|---:|---:|']
    keys=['recall_at1','recall_at5','recall_at10','recall_at50','group_recall_at1']
    for w in [[1,1,1]]:
        rs=[r for r in old if r['ratio']==w]
        lines.append(f'| (none) | {w} | '+' | '.join(f'{sum(r[k] for r in rs)/len(rs):.2f}' for k in keys)+' |')
        for c in combined:
            if c['ratio']==w:lines.append(f"| {c['prompt']} | {w} | "+' | '.join(f'{c[k]:.2f}' for k in keys)+' |')
    best=max(combined,key=lambda c:c['recall_at1'])
    lines+=['',f"Best prompt at 1/1/1: `{best['prompt']}`, R@1 {best['recall_at1']:.2f}."]
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)
    export_test(texts,best['prompt'],[[1,1,1]])
