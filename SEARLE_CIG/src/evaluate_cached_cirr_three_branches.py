"""Encode captions once and evaluate three branches using existing CIRR caches."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import clip
from cache_cirr_query_embeddings import load_reference_captions, compose_text_prompt, file_sha256
from evaluate_cirr_query_fusions import fuse, fusion_names, evaluate_fusion

@torch.no_grad()
def main():
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[2]
    out = root / 'analysis'
    triplets = json.loads((root / 'data/cirr/captions/cap.rc2.val.json').read_text())
    captions, path = load_reference_captions(root / 'data', 'val')
    ids = [int(t['pairid']) for t in triplets]
    prompts = [compose_text_prompt(captions[int(t['pairid'])], t['caption']) for t in triplets]
    model, _ = clip.load(str(root / 'ViT-B-32.pt'), device='cuda', jit=False)
    model = model.float().eval()
    text = torch.cat([F.normalize(model.encode_text(clip.tokenize(prompts[i:i+64], truncate=True).cuda()), dim=-1).cpu() for i in range(0, len(prompts), 64)])
    print('Encoded caption queries:', tuple(text.shape), flush=True)
    del model
    for seed in [1600, 1601]:
        source = out / f'query_agreement_seed{seed}.npz'
        with np.load(source, allow_pickle=False) as old:
            assert old['pids'].tolist() == ids
            assert old['refs'].tolist() == [t['reference'] for t in triplets]
            assert old['targets'].tolist() == [t['target_hard'] for t in triplets]
            cache = {'schema_version': 1, 'metadata': {'dataset': 'cirr', 'split': 'val', 'clip_model_name': 'ViT-B/32', 'eval_type': 'searle', 'preprocess_type': 'targetpad', 'seed': seed, 'source_cache': str(source), 'reference_caption_file': str(path), 'reference_caption_sha256': file_sha256(path), 'text_prompt': 'reference caption + delta caption'}, 'pair_ids': ids, 'ref': torch.from_numpy(old['q'].copy()), 'gen': torch.from_numpy(old['qg'].copy()), 'text': text, 'text_prompts': prompts, 'index': torch.from_numpy(old['index'].copy()), 'index_names': old['index_names'].tolist()}
        for key in ['ref', 'gen', 'text', 'index']:
            assert torch.isfinite(cache[key]).all() and cache[key].shape[1] == 512
        cache_path = out / f'cirr_three_branch_cache_seed{seed}.pt'
        torch.save(cache, cache_path)
        index = F.normalize(cache['index'].cuda().float(), dim=-1)
        results = {'cache_path': str(cache_path), 'cache_metadata': cache['metadata'], 'query_count': len(ids), 'candidate_count': len(index), 'fusions': {}}
        for name in fusion_names():
            metrics = evaluate_fusion(fuse(cache, name), index, cache['index_names'], triplets)
            results['fusions'][name] = metrics
            print(seed, name, metrics, flush=True)
        (out / f'cirr_three_branch_results_seed{seed}.json').write_text(json.dumps(results, indent=2))
        order = torch.argsort(fuse(cache, 'ref+gen+text').cuda() @ index.T, dim=-1, descending=True).cpu().numpy()
        predictions = {}
        for t, row in zip(triplets, np.asarray(cache['index_names'])[order]):
            predictions[str(t['pairid'])] = row[row != t['reference']][:50].tolist()
        (out / f'cirr_three_branch_top50_seed{seed}.json').write_text(json.dumps(predictions))

if __name__ == '__main__':
    main()
