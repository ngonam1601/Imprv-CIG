"""Plain reference-image captions with Idefics2-8B (no edit in the prompt); ablation for the edit-aware run.

Captions each distinct reference image once, then writes a per-query file captions_query_{split}.json whose
text is the reference caption, so dual_softmax_cirr.py builds "{caption} {delta}" exactly as for the
edit-aware descriptions. Writes only under analysis/idefics2_refcap/.
Usage: experiment_idefics2_refcap_cirr.py, then
       dual_softmax_cirr.py --caption-dir analysis/idefics2_refcap --weights 1 1 1
"""
import json
import re
import time
from PIL import Image
import torch
from transformers import AutoProcessor, Idefics2ForConditionalGeneration
from experiment_blip_cirr import ROOT, load_split
from experiment_idefics2_cirr import MODEL, REVISION, GENERATION

OUT=ROOT/'analysis/idefics2_refcap'
INSTRUCTION='Describe this image in one sentence.'
PROMPT=f'User:<image>{INSTRUCTION}<end_of_utterance>\nAssistant:'
# Idefics2 answers in the localized-narratives style ("In this image I can see ..."); keep only the content.
META_SENTENCE=re.compile(r'^(this|the) (image|picture|photo|pic)\b.*\b(taken|clicked|captured|edited)\b',re.I)
FRAMING=[re.compile(p,re.I) for p in [
    r'\b(in|on|at|to) (the )?(this |the )?((left|right|top|bottom|middle|center|centre|front|background|backside|foreground|corner)[a-z ]*? )?(side |part |corner )?(of )?(the |this )?(image|picture|photo|pic)\b,?\s*',
    r'\b(in|at|on) the (background|backside|foreground|front|middle|center|centre|bottom|top|left|right)( side| corner)?\b,?\s*',
    r'\b(i|we) (can |could )?(also )?(see|observe|notice)( that)?\b\s*',
    r'^(there (is|are)|it is|this is|these are)\s+',
]]

def clean(caption):
    sentences=[s.strip() for s in re.split(r'(?<=[.!?])\s+',caption.strip()) if s.strip()]
    kept=[]
    for sentence in sentences:
        if META_SENTENCE.search(sentence):continue
        for pattern in FRAMING:sentence=pattern.sub('',sentence).strip(' ,')
        if len(sentence.strip(' .'))>1:kept.append(sentence.rstrip('.'))
    return ', '.join(kept)

@torch.no_grad()
def generate_captions(batch_size=4):
    processor=AutoProcessor.from_pretrained(MODEL,revision=REVISION,do_image_splitting=False,size={'longest_edge':490,'shortest_edge':378})
    processor.tokenizer.padding_side='left'
    model=Idefics2ForConditionalGeneration.from_pretrained(MODEL,revision=REVISION,torch_dtype=torch.float16).cuda().eval()
    for split in ['val','test1']:
        triplets,paths=load_split(split)
        dest=OUT/f'captions_ref_{split}.json'
        saved=json.loads(dest.read_text()) if dest.exists() else {'model':MODEL,'revision':REVISION,'instruction':INSTRUCTION,'generation':GENERATION,'captions':{}}
        pending=[n for n in dict.fromkeys(t['reference'] for t in triplets) if n not in saved['captions']]
        print(split,'pending',len(pending),flush=True)
        start_time=time.time()
        for start in range(0,len(pending),batch_size):
            batch=pending[start:start+batch_size]
            images=[]
            for name in batch:
                with Image.open(ROOT/'data'/paths[name]) as img:images.append([img.convert('RGB')])
            inputs=processor(text=[PROMPT]*len(batch),images=images,return_tensors='pt',padding=True)
            inputs={k:v.to('cuda') for k,v in inputs.items()}
            inputs['pixel_values']=inputs['pixel_values'].half()
            tokens=model.generate(**inputs,**GENERATION)[:,inputs['input_ids'].shape[1]:]
            for name,caption in zip(batch,processor.batch_decode(tokens,skip_special_tokens=True)):
                assert caption.strip(),f'Empty caption {name}'
                saved['captions'][name]=caption.strip()
            temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(saved,indent=2));temp.replace(dest)
            done=start+len(batch)
            if done%200<batch_size:print(split,done,'/',len(pending),f'{(time.time()-start_time)/done:.2f}s/image',flush=True)
        # Same cleaned caption for every query sharing a reference image; raw captions stay in captions_ref_*.json.
        per_query={'source':str(dest),'cleaning':'framing phrases removed (clean())','captions':{str(t['pairid']):clean(saved['captions'][t['reference']]) or saved['captions'][t['reference']] for t in triplets}}
        (OUT/f'captions_query_{split}.json').write_text(json.dumps(per_query,indent=2))
    del model;torch.cuda.empty_cache()

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    generate_captions()
