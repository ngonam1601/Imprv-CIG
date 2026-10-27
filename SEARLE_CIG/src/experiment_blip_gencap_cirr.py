"""Caption the generated pseudo-target (not the reference) with BLIP; writes only under analysis/blip_gencap/.

The delta can remove or replace the reference content, so a reference caption can
contradict the target. The generated image already reflects the delta.
"""
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import clip
from transformers import BlipProcessor, BlipForConditionalGeneration
from experiment_blip_cirr import ROOT, MODEL, load_split, rank_setup, rank

OUT=ROOT/'analysis/blip_gencap'
# (split, seed, generated-image dir, cache with ref/gen/index branches)
SOURCES=[('val',1600,'data/generated_images_val','analysis/blip/cache_val_seed1600.pt'),
         ('val',1601,'data/generated_images_val_seed1601','analysis/blip/cache_val_seed1601.pt'),
         ('test1',1600,'data/generated_images_test1_seed1600','analysis/blip/cache_test1_seed1600.pt')]
VARIANTS={'gencap':lambda cap,delta:f"a photo of {cap.strip(' .')}",
          'gencap_delta':lambda cap,delta:f"a photo of {cap.strip(' .')}. {delta.strip(' .')}"}

@torch.no_grad()
def generate_captions():
    processor=BlipProcessor.from_pretrained(MODEL,cache_dir=ROOT/'data/hf_cache_blip')
    model=BlipForConditionalGeneration.from_pretrained(MODEL,cache_dir=ROOT/'data/hf_cache_blip',torch_dtype=torch.float16).cuda().eval()
    for split,seed,image_dir,_ in SOURCES:
        triplets,_=load_split(split)
        dest=OUT/f'captions_gen_{split}_seed{seed}.json'
        saved=json.loads(dest.read_text()) if dest.exists() else {'model':MODEL,'image_dir':image_dir,'generation':{'num_beams':3,'max_new_tokens':32,'do_sample':False},'captions':{}}
        pending=[str(t['pairid']) for t in triplets if str(t['pairid']) not in saved['captions']]
        print(split,seed,'pending',len(pending),flush=True)
        for start in range(0,len(pending),32):
            batch=pending[start:start+32]
            images=[]
            for pid in batch:
                with Image.open(ROOT/image_dir/f'{int(pid):05d}.png') as img:images.append(img.convert('RGB'))
            inputs=processor(images=images,return_tensors='pt').to('cuda',torch.float16)
            tokens=model.generate(**inputs,num_beams=3,max_new_tokens=32,do_sample=False)
            for pid,caption in zip(batch,processor.batch_decode(tokens,skip_special_tokens=True)):
                assert caption.strip(),f'Empty BLIP caption {pid}'
                saved['captions'][pid]=caption.strip()
            temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(saved,indent=2));temp.replace(dest)
    del model;torch.cuda.empty_cache()

@torch.no_grad()
def encode_captions():
    """Return {(split, seed): {variant: text embeddings}}."""
    model,_=clip.load(str(ROOT/'ViT-B-32.pt'),device='cuda',jit=False)
    model=model.float().eval();texts={}
    for split,seed,_,_ in SOURCES:
        triplets,_=load_split(split)
        captions=json.loads((OUT/f'captions_gen_{split}_seed{seed}.json').read_text())['captions']
        texts[split,seed]={}
        for variant,compose in VARIANTS.items():
            prompts=[compose(captions[str(t['pairid'])],t['caption']) for t in triplets]
            texts[split,seed][variant]=torch.cat([F.normalize(model.encode_text(clip.tokenize(prompts[i:i+64],truncate=True).cuda()),dim=-1).cpu() for i in range(0,len(prompts),64)])
    del model;torch.cuda.empty_cache()
    return texts

def evaluate(texts):
    ratios=[[1,1,1]]
    triplets,_=load_split('val');results=[]
    for split,seed,_,cache_path in SOURCES:
        if split!='val':continue
        cache=torch.load(ROOT/cache_path,map_location='cpu')
        for variant in VARIANTS:
            cache['text']=texts[split,seed][variant]
            names,lookup,refs,branches,index=rank_setup(cache,triplets)
            targets=torch.tensor([lookup[t['target_hard']] for t in triplets],device='cuda')
            groups=torch.zeros((len(triplets),len(names)),dtype=torch.bool,device='cuda')
            for i,t in enumerate(triplets):groups[i,[lookup[n] for n in t['img_set']['members']]]=True
            groups[torch.arange(len(refs),device='cuda'),refs]=False
            for w in ratios:
                order=rank(w,refs,branches,index)
                ranks=(order==targets[:,None]).int().argmax(dim=1)
                gr=(groups.gather(1,order)&(torch.arange(len(names),device='cuda')[None,:]<ranks[:,None])).sum(dim=1)
                results.append({'variant':variant,'seed':seed,'ratio':w,**{f'recall_at{k}':(ranks<k).float().mean().item()*100 for k in [1,5,10,50]},**{f'group_recall_at{k}':(gr<k).float().mean().item()*100 for k in [1,2,3]}})
            print('Evaluated',seed,variant,flush=True)
    (OUT/'validation_weight_sweep.json').write_text(json.dumps({'caption_model':MODEL,'variants':list(VARIANTS),'branch_order':['ref','gen','caption'],'results':results},indent=2))
    keys=['recall_at1','recall_at5','recall_at10','recall_at50','group_recall_at1']
    combined=[]
    for variant in VARIANTS:
        for w in ratios:
            rs=[r for r in results if r['variant']==variant and r['ratio']==w]
            combined.append({'variant':variant,'ratio':w,**{k:sum(r[k] for r in rs)/len(rs) for k in keys}})
    return combined

@torch.no_grad()
def export_test(texts,jobs):
    triplets,_=load_split('test1')
    cache=torch.load(ROOT/'analysis/blip/cache_test1_seed1600.pt',map_location='cpu')
    dest=OUT/'test1_seed1600_submissions';dest.mkdir(exist_ok=True)
    manifest={'caption_model':MODEL,'caption_source':'generated image','branch_order':['ref','gen','caption'],'submissions':[]}
    for variant,w in jobs:
        cache['text']=texts['test1',1600][variant]
        names,lookup,refs,branches,index=rank_setup(cache,triplets)
        names=np.asarray(names)
        order=rank(w,refs,branches,index).cpu().numpy()
        full={'version':'rc2','metric':'recall'};subset={'version':'rc2','metric':'recall_subset'}
        for t,row in zip(triplets,names[order]):
            key=str(t['pairid']);full[key]=row[:50].tolist()
            group=set(t['img_set']['members'])-{t['reference']}
            subset[key]=[name for name in row if name in group][:3]
            assert len(set(full[key]))==50 and t['reference'] not in full[key]
            assert len(set(subset[key]))==3
        label='_'.join(f'{a:g}'.replace('.','p') for a in w)
        name=f'cirr_test1_seed1600_blip_{variant}_{label}.json'
        (dest/name).write_text(json.dumps(full));(dest/('subset_'+name)).write_text(json.dumps(subset))
        manifest['submissions'].append({'variant':variant,'ratio':w,'full':name,'subset':'subset_'+name})
        print('Exported',variant,w,flush=True)
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    generate_captions();texts=encode_captions();combined=evaluate(texts)
    keys=['recall_at1','recall_at5','recall_at10','recall_at50','group_recall_at1']
    old=json.loads((ROOT/'analysis/blip/validation_weight_sweep.json').read_text())['results']
    lines=['# BLIP captions of generated images','',f'Model {MODEL}, beam 3, max 32 new tokens, caption of the CIG-XL image of each query. gencap = "a photo of {{cap}}"; gencap_delta = "a photo of {{cap}}. {{delta}}". refcap = earlier BLIP reference caption + delta (analysis/blip). Mean of validation seeds 1600/1601.','','| Caption | Ratio | R@1 | R@5 | R@10 | R@50 | Group R@1 |','|---|---|---:|---:|---:|---:|---:|']
    for w in [[0.5,0.5,0],[1,1,1]]:
        rs=[r for r in old if r['ratio']==w]
        lines.append(f'| refcap | {w} | '+' | '.join(f'{sum(r[k] for r in rs)/len(rs):.2f}' for k in keys)+' |')
        for c in combined:
            if c['ratio']==w and w[2]>0:lines.append(f"| {c['variant']} | {w} | "+' | '.join(f'{c[k]:.2f}' for k in keys)+' |')
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)
    export_test(texts,[(v,[1,1,1]) for v in VARIANTS])
