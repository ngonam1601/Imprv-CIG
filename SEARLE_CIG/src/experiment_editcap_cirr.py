"""Edit-aware captions: an instruction VLM describes the reference image after applying the delta.

Writes only under analysis/editcap/. Ref/gen/index branches are reused from analysis/blip caches.
Usage: experiment_editcap_cirr.py [--pilot N]
"""
import argparse
import json
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import clip
from transformers import AutoProcessor, LlavaForConditionalGeneration
from experiment_blip_cirr import ROOT, load_split, rank_setup, rank

OUT=ROOT/'analysis/editcap'
MODEL='llava-hf/llava-1.5-7b-hf'
REVISION='05ae2434cbb430be33edcba0c5203e7023f785b7'  # pre-2024-06 files; later revisions need transformers>=4.44
INSTRUCTION='Describe the image after this change: {edit}. Do not mention removed objects.'
GENERATION={'max_new_tokens':77,'do_sample':False}
VARIANTS={'editcap':lambda desc,delta:desc,
          'editcap_delta':lambda desc,delta:f"{desc.strip()} {delta.strip(' .')}"}
VAL_CACHES={1600:'analysis/blip/cache_val_seed1600.pt',1601:'analysis/blip/cache_val_seed1601.pt'}

def chat(edit):
    return f"USER: <image>\n{INSTRUCTION.format(edit=edit.strip(' .'))} ASSISTANT:"

@torch.no_grad()
def generate_captions(splits,limit=None,batch_size=8):
    processor=AutoProcessor.from_pretrained(MODEL,revision=REVISION)
    processor.tokenizer.padding_side='left'
    model=LlavaForConditionalGeneration.from_pretrained(MODEL,revision=REVISION,torch_dtype=torch.float16).cuda().eval()
    for split in splits:
        triplets,paths=load_split(split)
        triplets=triplets[:limit] if limit else triplets
        dest=OUT/(f'pilot_{split}.json' if limit else f'captions_edit_{split}.json')
        saved=json.loads(dest.read_text()) if dest.exists() else {'model':MODEL,'revision':REVISION,'instruction':INSTRUCTION,'generation':GENERATION,'captions':{}}
        pending=[t for t in triplets if str(t['pairid']) not in saved['captions']]
        print(split,'pending',len(pending),flush=True)
        start_time=time.time()
        for start in range(0,len(pending),batch_size):
            batch=pending[start:start+batch_size]
            images=[]
            for t in batch:
                with Image.open(ROOT/'data'/paths[t['reference']]) as img:images.append(img.convert('RGB'))
            inputs=processor(text=[chat(t['caption']) for t in batch],images=images,return_tensors='pt',padding=True).to('cuda',torch.float16)
            tokens=model.generate(**inputs,**GENERATION)[:,inputs['input_ids'].shape[1]:]
            for t,image,desc in zip(batch,images,processor.batch_decode(tokens,skip_special_tokens=True)):
                if not desc.strip():
                    # Batched fp16 generation occasionally returns EOS at once; redo this query alone.
                    single=processor(text=[chat(t['caption'])],images=[image],return_tensors='pt').to('cuda',torch.float16)
                    desc=processor.batch_decode(model.generate(**single,**GENERATION)[:,single['input_ids'].shape[1]:],skip_special_tokens=True)[0]
                    print('Regenerated alone',t['pairid'],flush=True)
                assert desc.strip(),f"Empty description {t['pairid']}"
                saved['captions'][str(t['pairid'])]=desc.strip()
            temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(saved,indent=2));temp.replace(dest)
            done=start+len(batch)
            if done%200<batch_size or limit:
                print(split,done,'/',len(pending),f'{(time.time()-start_time)/done:.2f}s/query',flush=True)
        if limit:
            for t in triplets:print(f"\n[{t['pairid']}] edit: {t['caption']}\n  -> {saved['captions'][str(t['pairid'])]}")
    del model;torch.cuda.empty_cache()

@torch.no_grad()
def encode_captions(splits=('val','test1')):
    model,_=clip.load(str(ROOT/'ViT-B-32.pt'),device='cuda',jit=False)
    model=model.float().eval();texts={}
    for split in splits:
        triplets,_=load_split(split)
        captions=json.loads((OUT/f'captions_edit_{split}.json').read_text())['captions']
        texts[split]={}
        for variant,compose in VARIANTS.items():
            prompts=[compose(captions[str(t['pairid'])],t['caption']) for t in triplets]
            texts[split][variant]=torch.cat([F.normalize(model.encode_text(clip.tokenize(prompts[i:i+64],truncate=True).cuda()),dim=-1).cpu() for i in range(0,len(prompts),64)])
    del model;torch.cuda.empty_cache()
    return texts

def evaluate(texts,w=(1,1,1)):
    triplets,_=load_split('val');results=[]
    for seed,cache_path in VAL_CACHES.items():
        cache=torch.load(ROOT/cache_path,map_location='cpu')
        for variant in VARIANTS:
            cache['text']=texts['val'][variant]
            names,lookup,refs,branches,index=rank_setup(cache,triplets)
            targets=torch.tensor([lookup[t['target_hard']] for t in triplets],device='cuda')
            groups=torch.zeros((len(triplets),len(names)),dtype=torch.bool,device='cuda')
            for i,t in enumerate(triplets):groups[i,[lookup[n] for n in t['img_set']['members']]]=True
            groups[torch.arange(len(refs),device='cuda'),refs]=False
            order=rank(list(w),refs,branches,index)
            ranks=(order==targets[:,None]).int().argmax(dim=1)
            gr=(groups.gather(1,order)&(torch.arange(len(names),device='cuda')[None,:]<ranks[:,None])).sum(dim=1)
            results.append({'variant':variant,'seed':seed,'ratio':list(w),**{f'recall_at{k}':(ranks<k).float().mean().item()*100 for k in [1,5,10,50]},**{f'group_recall_at{k}':(gr<k).float().mean().item()*100 for k in [1,2,3]}})
    (OUT/'validation_results.json').write_text(json.dumps({'model':MODEL,'revision':REVISION,'instruction':INSTRUCTION,'branch_order':['ref','gen','caption'],'results':results},indent=2))
    return results

@torch.no_grad()
def export_test(texts,w=(1,1,1)):
    triplets,_=load_split('test1')
    cache=torch.load(ROOT/'analysis/blip/cache_test1_seed1600.pt',map_location='cpu')
    dest=OUT/'test1_seed1600_submissions';dest.mkdir(exist_ok=True)
    manifest={'model':MODEL,'revision':REVISION,'instruction':INSTRUCTION,'branch_order':['ref','gen','caption'],'submissions':[]}
    for variant in VARIANTS:
        cache['text']=texts['test1'][variant]
        names,lookup,refs,branches,index=rank_setup(cache,triplets)
        names=np.asarray(names)
        order=rank(list(w),refs,branches,index).cpu().numpy()
        full={'version':'rc2','metric':'recall'};subset={'version':'rc2','metric':'recall_subset'}
        for t,row in zip(triplets,names[order]):
            key=str(t['pairid']);full[key]=row[:50].tolist()
            group=set(t['img_set']['members'])-{t['reference']}
            subset[key]=[name for name in row if name in group][:3]
            assert len(set(full[key]))==50 and t['reference'] not in full[key]
            assert len(set(subset[key]))==3
        label='_'.join(f'{a:g}'.replace('.','p') for a in w)
        name=f'cirr_test1_seed1600_{variant}_{label}.json'
        (dest/name).write_text(json.dumps(full));(dest/('subset_'+name)).write_text(json.dumps(subset))
        manifest['submissions'].append({'variant':variant,'ratio':list(w),'full':name,'subset':'subset_'+name})
        print('Exported',variant,list(w),flush=True)
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot',type=int,help='Caption only the first N validation queries and print them')
    parser.add_argument('--val-only',action='store_true',help='Caption and evaluate validation only; skip test1')
    args=parser.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    if args.pilot:
        generate_captions(['val'],limit=args.pilot)
    else:
        splits=['val'] if args.val_only else ['val','test1']
        generate_captions(splits)
        texts=encode_captions(splits);results=evaluate(texts)
        keys=['recall_at1','recall_at5','recall_at10','recall_at50','group_recall_at1']
        lines=['# Edit-aware captions (LLaVA-1.5-7B)','',f'Instruction: `{INSTRUCTION}` with the reference image. Greedy, max {GENERATION["max_new_tokens"]} new tokens. editcap = description; editcap_delta = description + delta. Ratio ref/gen/caption 1/1/1. Mean of validation seeds 1600/1601.','','| Caption | R@1 | R@5 | R@10 | R@50 | Group R@1 |','|---|---:|---:|---:|---:|---:|']
        for variant in VARIANTS:
            rs=[r for r in results if r['variant']==variant]
            lines.append(f'| {variant} | '+' | '.join(f'{sum(r[k] for r in rs)/len(rs):.2f}' for k in keys)+' |')
        (OUT/'report.md').write_text('\n'.join(lines)+'\n')
        print('\n'.join(lines),flush=True)
        if not args.val_only:export_test(texts)
