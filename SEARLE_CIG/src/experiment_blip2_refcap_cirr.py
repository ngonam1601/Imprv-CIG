"""BLIP-2 (OPT-2.7B) reference-image captions in place of BLIP for the src4-style caption branch.

Captions each distinct reference image once, without a prompt, greedy with max 50 new tokens as in the
dangkhoa-nguyen src4 BLIP code, and writes per-query captions_query_{split}.json for
dual_softmax_cirr.py --compose src4. Weights cache to /root (/workspace is over quota); outputs go to
analysis/blip2_refcap/.
"""
import json
import time
from pathlib import Path
from PIL import Image
import torch
from transformers import Blip2ForConditionalGeneration, Blip2Processor
from experiment_blip_cirr import ROOT, load_split

MODEL='Salesforce/blip2-opt-2.7b'
REVISION='51572668da0eb669e01a189dc22abe6088589a24'  # 2024-03 files with safetensors; later tokenizer.json needs newer tokenizers
CACHE_DIR=Path('/root/hf_cache_blip2')
OUT=ROOT/'analysis/blip2_refcap'
GENERATION={'max_new_tokens':50,'do_sample':False}

@torch.no_grad()
def generate_captions(batch_size=32):
    processor=Blip2Processor.from_pretrained(MODEL,revision=REVISION,cache_dir=CACHE_DIR)
    model=Blip2ForConditionalGeneration.from_pretrained(MODEL,revision=REVISION,cache_dir=CACHE_DIR,torch_dtype=torch.float16,use_safetensors=True).cuda().eval()
    for split in ['val','test1']:
        triplets,paths=load_split(split)
        dest=OUT/f'captions_ref_{split}.json'
        saved=json.loads(dest.read_text()) if dest.exists() else {'model':MODEL,'revision':REVISION,'prompt':None,'generation':GENERATION,'captions':{}}
        pending=[n for n in dict.fromkeys(t['reference'] for t in triplets) if n not in saved['captions']]
        print(split,'pending',len(pending),flush=True)
        start_time=time.time()
        for start in range(0,len(pending),batch_size):
            batch=pending[start:start+batch_size]
            images=[]
            for name in batch:
                with Image.open(ROOT/'data'/paths[name]) as img:images.append(img.convert('RGB'))
            inputs=processor(images=images,return_tensors='pt').to('cuda',torch.float16)
            for name,caption in zip(batch,processor.batch_decode(model.generate(**inputs,**GENERATION),skip_special_tokens=True)):
                assert caption.strip(),f'Empty caption {name}'
                saved['captions'][name]=caption.strip()
            temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(saved,indent=2));temp.replace(dest)
            done=start+len(batch)
            if done%320<batch_size:print(split,done,'/',len(pending),f'{(time.time()-start_time)/done:.3f}s/image',flush=True)
        per_query={'source':str(dest),'captions':{str(t['pairid']):saved['captions'][t['reference']] for t in triplets}}
        (OUT/f'captions_query_{split}.json').write_text(json.dumps(per_query,indent=2))
    del model;torch.cuda.empty_cache()

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    generate_captions()
