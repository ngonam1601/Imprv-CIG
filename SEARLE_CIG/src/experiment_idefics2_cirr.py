"""Edit-aware captions with Idefics2-8B (trained on The Cauldron, which includes NLVR2 train).

Same instruction, text variants and 1/1/1 evaluation as experiment_editcap_cirr.py; writes only under
analysis/idefics2/. Usage: experiment_idefics2_cirr.py [--pilot N] [--val-only]
"""
import argparse
import json
import time
from PIL import Image
import torch
from transformers import AutoProcessor, Idefics2ForConditionalGeneration
import experiment_editcap_cirr as editcap
from experiment_blip_cirr import ROOT, load_split

MODEL='HuggingFaceM4/idefics2-8b'
REVISION='2c031da2dc71f3ac989f9efa9b8ff476df3842c0'  # 2024-05 files; matches transformers 4.40
GENERATION={'max_new_tokens':77,'do_sample':False}
# Same instruction as the LLaVA run (experiment_editcap_cirr.INSTRUCTION) so the two models are comparable.
INSTRUCTION='Describe the image after this change: {edit}. Do not mention removed objects.'
assert INSTRUCTION==editcap.INSTRUCTION

def chat(edit):
    return f"User:<image>{INSTRUCTION.format(edit=edit.strip(' .'))}<end_of_utterance>\nAssistant:"

@torch.no_grad()
def generate_captions(splits,limit=None,batch_size=4):
    processor=AutoProcessor.from_pretrained(MODEL,revision=REVISION,do_image_splitting=False,size={'longest_edge':490,'shortest_edge':378})
    processor.tokenizer.padding_side='left'
    model=Idefics2ForConditionalGeneration.from_pretrained(MODEL,revision=REVISION,torch_dtype=torch.float16).cuda().eval()
    for split in splits:
        triplets,paths=load_split(split)
        triplets=triplets[:limit] if limit else triplets
        dest=editcap.OUT/(f'pilot_{split}.json' if limit else f'captions_edit_{split}.json')
        saved=json.loads(dest.read_text()) if dest.exists() else {'model':MODEL,'revision':REVISION,'instruction':INSTRUCTION,'generation':GENERATION,'captions':{}}
        pending=[t for t in triplets if str(t['pairid']) not in saved['captions']]
        print(split,'pending',len(pending),flush=True)
        start_time=time.time()
        for start in range(0,len(pending),batch_size):
            batch=pending[start:start+batch_size]
            images=[]
            for t in batch:
                with Image.open(ROOT/'data'/paths[t['reference']]) as img:images.append([img.convert('RGB')])
            inputs=processor(text=[chat(t['caption']) for t in batch],images=images,return_tensors='pt',padding=True)
            inputs={k:v.to('cuda') for k,v in inputs.items()}
            inputs['pixel_values']=inputs['pixel_values'].half()
            tokens=model.generate(**inputs,**GENERATION)[:,inputs['input_ids'].shape[1]:]
            for t,desc in zip(batch,processor.batch_decode(tokens,skip_special_tokens=True)):
                assert desc.strip(),f"Empty description {t['pairid']}"
                saved['captions'][str(t['pairid'])]=desc.strip()
            temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(saved,indent=2));temp.replace(dest)
            done=start+len(batch)
            if done%200<batch_size or limit:
                print(split,done,'/',len(pending),f'{(time.time()-start_time)/done:.2f}s/query',flush=True)
        if limit:
            for t in triplets:print(f"\n[{t['pairid']}] edit: {t['caption']}\n  -> {saved['captions'][str(t['pairid'])]}")
    del model;torch.cuda.empty_cache()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot',type=int,help='Caption only the first N validation queries and print them')
    parser.add_argument('--val-only',action='store_true',help='Caption and evaluate validation only; skip test1')
    args=parser.parse_args()
    torch.set_num_threads(4)
    # Reuse the LLaVA pipeline (encoding, evaluation, export) with a separate output directory.
    editcap.OUT=ROOT/'analysis/idefics2'
    editcap.MODEL,editcap.REVISION=MODEL,REVISION
    editcap.OUT.mkdir(parents=True,exist_ok=True)
    if args.pilot:
        generate_captions(['val'],limit=args.pilot)
    else:
        splits=['val'] if args.val_only else ['val','test1']
        generate_captions(splits)
        texts=editcap.encode_captions(splits);results=editcap.evaluate(texts)
        keys=['recall_at1','recall_at5','recall_at10','recall_at50','group_recall_at1']
        lines=['# Edit-aware captions (Idefics2-8B)','',f'Instruction: `{INSTRUCTION}` with the reference image (longest edge 490, no image splitting). Greedy, max {GENERATION["max_new_tokens"]} new tokens. editcap = description; editcap_delta = description + delta. Ratio ref/gen/caption 1/1/1. Mean of validation seeds 1600/1601.','','| Caption | R@1 | R@5 | R@10 | R@50 | Group R@1 |','|---|---:|---:|---:|---:|---:|']
        for variant in editcap.VARIANTS:
            rs=[r for r in results if r['variant']==variant]
            lines.append(f'| {variant} | '+' | '.join(f'{sum(r[k] for r in rs)/len(rs):.2f}' for k in keys)+' |')
        (editcap.OUT/'report.md').write_text('\n'.join(lines)+'\n')
        print('\n'.join(lines),flush=True)
        if not args.val_only:editcap.export_test(texts)
