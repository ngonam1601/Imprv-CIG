"""Retrieval with LinCIR instead of SEARLE, same CIG-XL images and caption branch.

LinCIR evaluation as in navervision/lincir: HuggingFace CLIP (ViT-L/14 or ViT-G/14), Phi on the projected image
embedding, query "a photo of $ that {delta}" with "$" replaced by the pseudo token. Phi weights are the LinCIR ones
in pretrained_weights/ (also used by extract_lincir_feat.py). The caption branch is encoded with the same CLIP text
encoder. Weights, dual-softmax and export follow experiment_vitl14_cirr.py. Caches go to /root/lincir_cache/<backbone>;
reports and submissions to analysis/lincir_<backbone>/.
Usage: experiment_lincir_cirr.py --backbone L [--caption-dir analysis/qwen3vl8b_composed --compose only]
"""
import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
from data_utils import collate_fn
from datasets import CIRRDataset
from phi import Phi
from experiment_blip_cirr import ROOT, load_split
from dual_softmax_cirr import encode_captions, queries
from experiment_vitl14_cirr import SOURCES, WEIGHTS, TEMPERATURES, ranks, metrics, export

BACKBONES={'L':('openai/clip-vit-large-patch14','pretrained_weights/phi_best.pt',768,torch.float32,64),
           'G':('Geonmo/CLIP-Giga-config-fixed','pretrained_weights/phi_best_giga.pt',1280,torch.float16,32)}
PROCESSOR=CLIPImageProcessor(crop_size={'height':224,'width':224},do_center_crop=True,do_convert_rgb=True,do_normalize=True,
                             do_rescale=True,do_resize=True,image_mean=[0.48145466,0.4578275,0.40821073],
                             image_std=[0.26862954,0.26130258,0.27577711],resample=3,size={'shortest_edge':224})

def preprocess(image):
    return PROCESSOR(images=image,return_tensors='pt')['pixel_values'][0]

def encode_with_pseudo_tokens(text_model,text,pseudo_tokens):
    """Pooled text embedding with token id 259 ("$") replaced by the pseudo token (LinCIR)."""
    embeddings=text_model.text_model.embeddings
    x=embeddings.token_embedding(text)
    x=torch.where(text.unsqueeze(-1)==259,pseudo_tokens.unsqueeze(1).to(x.dtype),x)
    x=x+embeddings.position_embedding(embeddings.position_ids)
    mask=torch.full((text.shape[1],text.shape[1]),torch.finfo(x.dtype).min,device=x.device).triu(1).to(x.dtype)
    mask=mask[None,None].expand(len(text),1,-1,-1)
    x=text_model.text_model.final_layer_norm(text_model.text_model.encoder(inputs_embeds=x,causal_attention_mask=mask)[0])
    return text_model.text_projection(x[torch.arange(len(x)),text.argmax(dim=-1)])

class TextEncoder(nn.Module):
    """clip.model.CLIP-style encode_text for encode_captions."""
    def __init__(self,text_model):
        super().__init__();self.text_model=text_model
    def encode_text(self,text):
        return self.text_model(input_ids=text).text_embeds.float()

@torch.no_grad()
def build_caches(backbone,cache):
    name,phi_path,dim,dtype,batch_size=BACKBONES[backbone]
    paths={key:cache/f'cache_{key[0]}_seed{key[1]}.pt' for key in SOURCES}
    if all(p.exists() for p in paths.values()):return
    vision=CLIPVisionModelWithProjection.from_pretrained(name,torch_dtype=dtype,cache_dir=str(ROOT)).cuda().eval()
    text=CLIPTextModelWithProjection.from_pretrained(name,torch_dtype=dtype,cache_dir=str(ROOT)).cuda().eval()
    phi=Phi(input_dim=dim,hidden_dim=dim*4,output_dim=dim,dropout=0)
    phi.load_state_dict(torch.load(ROOT/phi_path,map_location='cpu')['Phi']);phi=phi.cuda().eval()
    def image_embeds(images):
        return vision(pixel_values=images.cuda().to(dtype)).image_embeds.float()
    index={}
    for split in ['val','test1']:
        loader=DataLoader(CIRRDataset(ROOT/'data',split,'classic',preprocess),batch_size=batch_size,num_workers=4,collate_fn=collate_fn)
        features,names=[],[]
        for batch in tqdm(loader,desc=f'{split} index'):
            features.append(F.normalize(image_embeds(batch['image']),dim=-1).cpu());names+=batch['image_name']
        index[split]=(torch.cat(features),names)
    for (split,seed),gen_dir in SOURCES.items():
        path=paths[(split,seed)]
        if path.exists():continue
        triplets,_=load_split(split)
        loader=DataLoader(CIRRDataset(ROOT/'data',split,'relative',preprocess,generated_image_dir=ROOT/gen_dir),batch_size=batch_size,num_workers=4,collate_fn=collate_fn)
        ref,gen,pair_ids=[],[],[]
        for batch in tqdm(loader,desc=f'{split} seed {seed}'):
            tokens=clip.tokenize([f'a photo of $ that {c}' for c in batch['relative_caption']],truncate=True).cuda()
            ref.append(F.normalize(encode_with_pseudo_tokens(text,tokens,phi(image_embeds(batch['reference_image']))).float(),dim=-1).cpu())
            gen.append(F.normalize(encode_with_pseudo_tokens(text,tokens,phi(image_embeds(batch['generated_image']))).float(),dim=-1).cpu())
            pair_ids+=[int(p) for p in batch['pair_id']]
        assert pair_ids==[t['pairid'] for t in triplets]
        torch.save({'metadata':{'clip':name,'phi':phi_path,'preprocess':'clip center crop','generated_image_dir':gen_dir},
                    'pair_ids':pair_ids,'ref':torch.cat(ref),'gen':torch.cat(gen),'index':index[split][0],'index_names':index[split][1]},path)
        print('Saved',path,flush=True)

@torch.no_grad()
def caption_texts(backbone,cache,caption_dir,compose,tag):
    paths={split:cache/f'text_{tag}_{split}.pt' for split in ['val','test1']}
    if not all(p.exists() for p in paths.values()):
        name,_,_,dtype,_=BACKBONES[backbone]
        model=TextEncoder(CLIPTextModelWithProjection.from_pretrained(name,torch_dtype=dtype,cache_dir=str(ROOT)).cuda().eval())
        for split,path in paths.items():
            torch.save(encode_captions(caption_dir,split,load_split(split)[0],compose,model),path)
    return {split:torch.load(path) for split,path in paths.items()}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--backbone',choices=list(BACKBONES),default='L',help='L: ViT-L/14, G: ViT-G/14')
    parser.add_argument('--caption-dir',default='analysis/qwen3vl8b_composed')
    parser.add_argument('--compose',choices=['plain','src4','only'],default='only',help='As in dual_softmax_cirr.py')
    args=parser.parse_args()
    torch.set_num_threads(8)
    cache=Path(f'/root/lincir_cache/{args.backbone}');out=ROOT/f'analysis/lincir_{args.backbone}'
    cache.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    build_caches(args.backbone,cache)
    tag=args.caption_dir.rstrip('/').split('/')[-1]+('' if args.compose=='plain' else '_'+args.compose)
    texts=caption_texts(args.backbone,cache,args.caption_dir,args.compose,tag)
    val,_=load_split('val');test,_=load_split('test1')
    caches={seed:dict(torch.load(cache/f'cache_val_seed{seed}.pt',map_location='cpu'),text=texts['val']) for seed in [1600,1601]}
    test_cache=dict(torch.load(cache/'cache_test1_seed1600.pt',map_location='cpu'),text=texts['test1'])
    half_a=torch.tensor([t['pairid']%2==0 for t in val]);everything=torch.ones_like(half_a)
    # Temperature on the even half only, with the caption-free 1/1/0 query.
    tuning={T:sum(metrics(*ranks(caches[s],val,[1,1,0],T,None),half_a)['recall_at1'] for s in caches)/2 for T in TEMPERATURES}
    T=max(tuning,key=tuning.get);print('Half-A R@1 by temperature',tuning,'-> T',T,flush=True)
    keys=['recall_at1','recall_at5','recall_at10','recall_at50','group_recall_at1']
    backbone_name={'L':'ViT-L/14','G':'ViT-G/14'}[args.backbone]
    lines=[f'# LinCIR (CLIP {backbone_name}) on CIRR validation','',f'Same CIG-XL images as the SEARLE runs; caption branch {args.caption_dir} ({args.compose}). Dual-softmax T={T:g}, chosen on even pair ids (half-A R@1: '+', '.join(f'{k:g}: {v:.2f}' for k,v in tuning.items())+'). Full validation, mean of seeds 1600/1601. "own" = transductive over the split\'s own queries; "bank" = other split\'s queries (test1 for val, val seed 1600 for test1).','','| Ref/gen/caption | Scoring | R@1 | R@5 | R@10 | R@50 | Group R@1 |','|---|---|---:|---:|---:|---:|---:|']
    results=[]
    for w in WEIGHTS:
        label='_'.join(f'{a:g}'.replace('.','p') for a in w)
        val_bank=queries(test_cache,w);test_bank=queries(caches[1600],w)
        for mode,temperature,bank,test_bank_used,suffix in [('plain',None,None,None,''),('dsm own',T,None,None,f'_dsm_T{T:g}'),('dsm bank',T,val_bank,test_bank,f'_dsmbank_T{T:g}')]:
            rs=[metrics(*ranks(caches[s],val,w,temperature,bank),everything) for s in caches]
            mean={k:sum(r[k] for r in rs)/len(rs) for k in keys}
            results.append({'weights':w,'scoring':mode,'temperature':temperature,'per_seed':rs,'mean':mean})
            lines.append(f'| {"/".join(f"{a:g}" for a in w)} | {mode} | '+' | '.join(f'{mean[k]:.2f}' for k in keys)+' |')
            export(test_cache,test,w,temperature,test_bank_used,f"cirr_test1_seed1600_lincir{args.backbone}_{tag}_{label}{suffix.replace('.','p')}.json",out)
            print(lines[-1],flush=True)
    (out/f'validation_results_{tag}.json').write_text(json.dumps({'backbone':backbone_name,'caption':f'{args.caption_dir} ({args.compose})','temperature':T,'tuning':tuning,'results':results},indent=2))
    (out/f'report_{tag}.md').write_text('\n'.join(lines)+'\n')
    print('DONE',flush=True)
