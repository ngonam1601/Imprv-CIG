"""Cache test1 candidates and export weighted three-branch CIRR submissions."""
import json
from pathlib import Path
import clip
import numpy as np
import torch
import torch.nn.functional as F
from datasets import CIRRDataset
from data_utils import targetpad_transform
from utils_feat import extract_image_features

@torch.no_grad()
def main():
    torch.set_num_threads(4)
    root=Path(__file__).resolve().parents[2]
    path=root/'analysis/cirr_three_branch_cache_test1_seed1600.pt'
    cache=torch.load(path,map_location='cpu')
    triplets=json.loads((root/'data/cirr/captions/cap.rc2.test1.json').read_text())
    assert cache['pair_ids']==[int(t['pairid']) for t in triplets]
    if 'index' not in cache:
        model,_=clip.load(str(root/'ViT-B-32.pt'),device='cuda',jit=False)
        model=model.float().eval()
        ds=CIRRDataset(root/'data','test1','classic',targetpad_transform(1.25,model.visual.input_resolution))
        features,names=extract_image_features(ds,model,batch_size=64,num_workers=4)
        cache['index']=F.normalize(features.float(),dim=-1)
        cache['index_names']=names
        torch.save(cache,path)
        del model
    names=np.asarray(cache['index_names'])
    lookup={name:i for i,name in enumerate(names)}
    refs=torch.tensor([lookup[t['reference']] for t in triplets],device='cuda')
    rows=torch.arange(len(triplets),device='cuda')
    index=F.normalize(cache['index'].float().cuda(),dim=-1)
    branches=[F.normalize(cache[k].float().cuda(),dim=-1) for k in ['ref','gen','text']]
    sweep=json.loads((root/'analysis/cirr_three_branch_weight_sweep.json').read_text())
    ratios=[r['ratio'] for r in sweep['results'] if r['seed']==1600]
    dest=root/'analysis/test1_seed1600_submissions'
    dest.mkdir(exist_ok=True)
    manifest={'branch_order':['ref','gen','caption'],'queries':len(triplets),'candidates':len(names),'cache':str(path),'submissions':[]}
    for j,w in enumerate(ratios):
        query=F.normalize(sum(a*b for a,b in zip(w,branches)),dim=-1)
        scores=query@index.T
        scores[rows,refs]=-torch.inf
        order=torch.argsort(1-scores,dim=-1).cpu().numpy()
        full={'version':'rc2','metric':'recall'}
        subset={'version':'rc2','metric':'recall_subset'}
        for t,row in zip(triplets,names[order]):
            key=str(t['pairid'])
            prediction=row[:50].tolist()
            members=set(t['img_set']['members'])-{t['reference']}
            group=[name for name in row if name in members][:3]
            assert len(prediction)==50 and len(set(prediction))==50 and t['reference'] not in prediction
            assert len(group)==3 and len(set(group))==3
            full[key]=prediction;subset[key]=group
        label='_'.join(f'{v:g}'.replace('.','p') for v in w)
        name=f'cirr_test1_seed1600_{label}.json'
        (dest/name).write_text(json.dumps(full))
        (dest/('subset_'+name)).write_text(json.dumps(subset))
        manifest['submissions'].append({'ratio':w,'full':name,'subset':'subset_'+name})
        if j%10==0:print('Exported',j+1,'/',len(ratios),flush=True)
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('DONE',dest,flush=True)

if __name__=='__main__':main()
