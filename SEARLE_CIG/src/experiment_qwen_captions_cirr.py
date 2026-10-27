"""Qwen2.5-VL-3B captions for CIRR, in the two roles compared in the caption ablation.

--task refcap:   caption each distinct reference image ("Describe this image in one sentence."), as BLIP does
                 in dangkhoa-nguyen src4; writes analysis/qwen25vl_refcap/captions_query_{split}.json.
--task composed: describe the imagined target image with the src5 system prompt (prompt_src5.py) and the
                 src5 user text; writes analysis/qwen25vl_composed/captions_edit_{split}.json.
Evaluate with dual_softmax_cirr.py in the main venv. Needs transformers>=4.49, so it runs in its own venv:
    /root/venv_qwen/bin/python SEARLE_CIG/src/experiment_qwen_captions_cirr.py --task composed
"""
import argparse
import json
import re
import time
from pathlib import Path
from PIL import Image
import torch
import transformers
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from prompt_src5 import system_prompt as SRC5_SYSTEM_PROMPT

ROOT=Path(__file__).resolve().parents[2]
# src5 uses the AWQ / FP8 builds of these models; unquantised weights avoid autoawq and FP8 (needs Ada/Hopper).
# patch: vision patch size; each visual token covers (2*patch)^2 pixels after 2x2 merging.
MODELS={'qwen2.5-vl-3b':{'id':'Qwen/Qwen2.5-VL-3B-Instruct','cls':'Qwen2_5_VLForConditionalGeneration','patch':14,'out':'qwen25vl'},
        'qwen3-vl-8b':{'id':'Qwen/Qwen3-VL-8B-Instruct','cls':'Qwen3VLForConditionalGeneration','patch':16,'out':'qwen3vl8b'}}
CACHE_DIR=Path('/root/hf_cache_qwen')  # /workspace is over quota
VISUAL_TOKENS=512  # caps visual tokens per image to keep batches on a 24 GB GPU
REFCAP_INSTRUCTION='Describe this image in one sentence.'
SRC5_USER='MODIFICATION:\n{edit}\n\nGenerate the target-image description. Return only one concise sentence.'

def load_split(split):
    triplets=json.loads((ROOT/f'data/cirr/captions/cap.rc2.{split}.json').read_text())
    paths=json.loads((ROOT/f'data/cirr/image_splits/split.rc2.{split}.json').read_text())
    return triplets,paths

def messages(image,task,max_pixels,edit=None):
    if task=='refcap':
        return [{'role':'user','content':[{'type':'image','image':image,'max_pixels':max_pixels},{'type':'text','text':REFCAP_INSTRUCTION}]}]
    return [{'role':'system','content':SRC5_SYSTEM_PROMPT.strip()},
            {'role':'user','content':[{'type':'image','image':image,'max_pixels':max_pixels},{'type':'text','text':SRC5_USER.format(edit=edit.strip())}]}]

def clean(output):
    # Same post-processing as src5: drop reasoning blocks, control tokens and surrounding quotes.
    if '</think>' in output:output=output.rsplit('</think>',1)[-1]
    output=re.sub(r'<\|[^|]+\|>',' ',output)
    return ' '.join(output.split()).strip().strip('"').strip("'")

@torch.no_grad()
def run(model_key,task,splits,batch_size,limit):
    spec=MODELS[model_key];MODEL=spec['id'];qwen3=model_key.startswith('qwen3')
    MAX_PIXELS=VISUAL_TOKENS*(2*spec['patch'])**2
    out=ROOT/f"analysis/{spec['out']}_{task}"
    out.mkdir(parents=True,exist_ok=True)
    processor=AutoProcessor.from_pretrained(MODEL,cache_dir=CACHE_DIR,max_pixels=MAX_PIXELS)
    processor.tokenizer.padding_side='left'
    model=getattr(transformers,spec['cls']).from_pretrained(MODEL,cache_dir=CACHE_DIR,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    generation={'max_new_tokens':128 if task=='composed' else 77,'do_sample':False}
    for split in splits:
        triplets,paths=load_split(split)
        triplets=triplets[:limit] if limit else triplets
        name='captions_ref' if task=='refcap' else 'captions_edit'
        dest=out/(f'pilot_{split}.json' if limit else f'{name}_{split}.json')
        saved=json.loads(dest.read_text()) if dest.exists() else {'model':MODEL,'max_pixels':MAX_PIXELS,'generation':generation,
            'prompt':REFCAP_INSTRUCTION if task=='refcap' else {'system':'prompt_src5.system_prompt','user':SRC5_USER},'captions':{}}
        # refcap: one caption per distinct reference image; composed: one description per query.
        items=list(dict.fromkeys(t['reference'] for t in triplets)) if task=='refcap' else [t for t in triplets]
        key=(lambda x:x) if task=='refcap' else (lambda t:str(t['pairid']))
        pending=[x for x in items if key(x) not in saved['captions']]
        print(task,split,'pending',len(pending),flush=True)
        start_time=time.time()
        for start in range(0,len(pending),batch_size):
            batch=pending[start:start+batch_size]
            convs=[]
            for x in batch:
                ref=x if task=='refcap' else x['reference']
                with Image.open(ROOT/'data'/paths[ref]) as img:image=img.convert('RGB')
                convs.append(messages(image,task,MAX_PIXELS,None if task=='refcap' else x['caption']))
            texts=[processor.apply_chat_template(c,tokenize=False,add_generation_prompt=True) for c in convs]
            if qwen3:
                # As in src5: qwen-vl-utils resizes for 16-pixel patches, so the processor must not resize again.
                images,_=process_vision_info(convs,image_patch_size=16)
                inputs=processor(text=texts,images=images,padding=True,return_tensors='pt',do_resize=False).to('cuda')
            else:
                images,_=process_vision_info(convs)
                inputs=processor(text=texts,images=images,padding=True,return_tensors='pt').to('cuda')
            tokens=model.generate(**inputs,**generation)[:,inputs['input_ids'].shape[1]:]
            for x,output in zip(batch,processor.batch_decode(tokens,skip_special_tokens=True)):
                caption=clean(output)
                assert caption,f'Empty output for {key(x)}'
                saved['captions'][key(x)]=caption
            temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(saved,indent=2));temp.replace(dest)
            done=start+len(batch)
            if done%200<batch_size or limit:print(task,split,done,'/',len(pending),f'{(time.time()-start_time)/done:.2f}s/item',flush=True)
        if limit:
            for t in triplets:
                print(f"\n[{t['pairid']}] edit: {t['caption']}\n  -> {saved['captions'][key(t['reference'] if task=='refcap' else t)]}")
        elif task=='refcap':
            per_query={'source':str(dest),'captions':{str(t['pairid']):saved['captions'][t['reference']] for t in triplets}}
            (out/f'captions_query_{split}.json').write_text(json.dumps(per_query,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=list(MODELS),default='qwen2.5-vl-3b')
    parser.add_argument('--task',choices=['refcap','composed'],required=True)
    parser.add_argument('--splits',nargs='+',default=['val','test1'])
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--pilot',type=int,help='Only the first N validation queries; print the outputs')
    args=parser.parse_args()
    run(args.model,args.task,['val'] if args.pilot else args.splits,args.batch_size,args.pilot)
