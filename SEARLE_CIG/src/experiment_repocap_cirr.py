"""Caption branch exactly as dangkhoa-nguyen/ComposedImageGen src4 (CIRCO code), applied to CIRR.

BLIP-base fp32, greedy, max_new_tokens=50 on the reference image; text "{caption}, {delta}" lowercased;
clip.tokenize without truncation. Writes only under analysis/repocap/.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import clip
from transformers import BlipProcessor, BlipForConditionalGeneration
from experiment_blip_cirr import ROOT, MODEL, load_split, rank_setup, rank

OUT=ROOT/'analysis/repocap'
GENERATION={'max_new_tokens':50}
VAL_CACHES={1600:'analysis/blip/cache_val_seed1600.pt',1601:'analysis/blip/cache_val_seed1601.pt'}
WEIGHTS=[1,1,1]
CACHE_DIR=ROOT/'data/hf_cache_blip'

def compose(caption,delta):
    return f"{caption.strip().lower()}, {delta.strip().lower()}"

@torch.no_grad()
def generate_captions():
    processor=BlipProcessor.from_pretrained(MODEL,cache_dir=CACHE_DIR)
    model=BlipForConditionalGeneration.from_pretrained(MODEL,cache_dir=CACHE_DIR).cuda().eval()
    for split in ['val','test1']:
        triplets,paths=load_split(split)
        dest=OUT/f'captions_ref_{split}.json'
        saved=json.loads(dest.read_text()) if dest.exists() else {'model':MODEL,'generation':GENERATION,'captions':{}}
        # Greedy decoding is deterministic, so one caption per distinct reference equals per-query captioning.
        pending=[n for n in dict.fromkeys(t['reference'] for t in triplets) if n not in saved['captions']]
        print(split,'pending',len(pending),flush=True)
        for start in range(0,len(pending),32):
            batch=pending[start:start+32]
            images=[]
            for name in batch:
                with Image.open(ROOT/'data'/paths[name]) as img:images.append(img.convert('RGB'))
            inputs=processor(images=images,return_tensors='pt',padding=True).to('cuda')
            for name,caption in zip(batch,processor.batch_decode(model.generate(**inputs,**GENERATION),skip_special_tokens=True)):
                saved['captions'][name]=caption
            temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(saved,indent=2));temp.replace(dest)
    del model;torch.cuda.empty_cache()

@torch.no_grad()
def encode_captions():
    # Same weights as clip.load('ViT-B/32', download_root='./') in that repo.
    model,_=clip.load(str(ROOT/'ViT-B-32.pt'),device='cuda',jit=False)
    model=model.float().eval();texts={}
    for split in ['val','test1']:
        triplets,_=load_split(split)
        captions=json.loads((OUT/f'captions_ref_{split}.json').read_text())['captions']
        prompts=[compose(captions[t['reference']],t['caption']) for t in triplets]
        texts[split]=torch.cat([F.normalize(model.encode_text(clip.tokenize(prompts[i:i+64],context_length=77,truncate=True).cuda()),dim=-1).cpu() for i in range(0,len(prompts),64)])
        (OUT/f'composed_{split}.json').write_text(json.dumps({str(t['pairid']):p for t,p in zip(triplets,prompts)},indent=2))
    del model;torch.cuda.empty_cache()
    return texts

def evaluate(texts):
    triplets,_=load_split('val');results=[]
    for seed,cache_path in VAL_CACHES.items():
        cache=torch.load(ROOT/cache_path,map_location='cpu')
        cache['text']=texts['val']
        names,lookup,refs,branches,index=rank_setup(cache,triplets)
        targets=torch.tensor([lookup[t['target_hard']] for t in triplets],device='cuda')
        groups=torch.zeros((len(triplets),len(names)),dtype=torch.bool,device='cuda')
        for i,t in enumerate(triplets):groups[i,[lookup[n] for n in t['img_set']['members']]]=True
        groups[torch.arange(len(refs),device='cuda'),refs]=False
        order=rank(WEIGHTS,refs,branches,index)
        ranks=(order==targets[:,None]).int().argmax(dim=1)
        gr=(groups.gather(1,order)&(torch.arange(len(names),device='cuda')[None,:]<ranks[:,None])).sum(dim=1)
        results.append({'seed':seed,'ratio':WEIGHTS,**{f'recall_at{k}':(ranks<k).float().mean().item()*100 for k in [1,5,10,50]},**{f'group_recall_at{k}':(gr<k).float().mean().item()*100 for k in [1,2,3]}})
    (OUT/'validation_results.json').write_text(json.dumps({'model':MODEL,'generation':GENERATION,'text':'{caption}, {delta} lowercased','branch_order':['ref','gen','caption'],'results':results},indent=2))
    return results

@torch.no_grad()
def export_test(texts):
    triplets,_=load_split('test1')
    cache=torch.load(ROOT/'analysis/blip/cache_test1_seed1600.pt',map_location='cpu')
    cache['text']=texts['test1']
    names,lookup,refs,branches,index=rank_setup(cache,triplets)
    names=np.asarray(names);order=rank(WEIGHTS,refs,branches,index).cpu().numpy()
    full={'version':'rc2','metric':'recall'};subset={'version':'rc2','metric':'recall_subset'}
    for t,row in zip(triplets,names[order]):
        key=str(t['pairid']);full[key]=row[:50].tolist()
        group=set(t['img_set']['members'])-{t['reference']}
        subset[key]=[name for name in row if name in group][:3]
        assert len(set(full[key]))==50 and t['reference'] not in full[key]
        assert len(set(subset[key]))==3
    dest=OUT/'test1_seed1600_submissions';dest.mkdir(exist_ok=True)
    label='_'.join(f'{a:g}'.replace('.','p') for a in WEIGHTS)
    name=f'cirr_test1_seed1600_repocap_{label}.json'
    (dest/name).write_text(json.dumps(full));(dest/('subset_'+name)).write_text(json.dumps(subset))
    print('Exported',dest/name,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--no-repeat',action='store_true',help='Block repeated trigrams; writes to analysis/repocap_norep/')
    parser.add_argument('--large',action='store_true',help='BLIP-large with --no-repeat settings; writes to analysis/repocap_large/')
    args=parser.parse_args()
    if args.large:
        MODEL='Salesforce/blip-image-captioning-large'
        CACHE_DIR=Path('/root/hf_cache_blip_large')  # /workspace is over quota
        OUT=ROOT/'analysis/repocap_large'
        GENERATION.update(no_repeat_ngram_size=3,repetition_penalty=1.3)
    elif args.no_repeat:
        # Greedy BLIP loops on out-of-domain images ("staples staples ..."); block it.
        OUT=ROOT/'analysis/repocap_norep'
        GENERATION.update(no_repeat_ngram_size=3,repetition_penalty=1.3)
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    generate_captions();texts=encode_captions();results=evaluate(texts)
    keys=['recall_at1','recall_at5','recall_at10','recall_at50','group_recall_at1']
    lines=['# Repo-style caption branch (dangkhoa-nguyen src4) on CIRR','',f'Model {MODEL}. Generation {GENERATION}. BLIP fp32 greedy, reference image; text `{{caption}}, {{delta}}` lowercased; truncated to 77 tokens. Ratio ref/gen/caption 1/1/1.','','| Seed | R@1 | R@5 | R@10 | R@50 | Group R@1 |','|---|---:|---:|---:|---:|---:|']
    for r in results:lines.append(f"| {r['seed']} | "+' | '.join(f'{r[k]:.2f}' for k in keys)+' |')
    lines.append('| mean | '+' | '.join(f'{sum(r[k] for r in results)/len(results):.2f}' for k in keys)+' |')
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)
    export_test(texts)
