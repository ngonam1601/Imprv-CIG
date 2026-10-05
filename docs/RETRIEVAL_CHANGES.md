# Retrieval changes and experiment commands

These changes build on upstream `lan-lw/ComposedImageGen` commit `49d0be0`.
The order of fusion weights is **reference / generated / caption**. Each branch
is normalized, combined with its weight, then the query is normalized again.
Candidate images are ranked by cosine similarity; the reference is excluded.
Baseline weights are `0.5/0.5/0`.

## Existing files changed

| File | Change |
|---|---|
| `extract_lincir_feat.py` | Resolve Phi checkpoints under `pretrained_weights/` and CIRR under `data/cirr/`, relative to this repository. Reduce feature extraction batch size from 4 to 1 for GPU memory. |
| `test_CIG.py` | Load the SDXL VAE with `use_safetensors=True`. |
| `SEARLE_CIG/src/validate.py` | Add CIRR reference-caption loading and a third query branch; expose three nonnegative fusion weights, defaulting to thirds. Pass the generated-image directory to the relative CIRR dataset. Return target names rather than pair IDs for validation ground truth. The pre-existing local FashionIQ fusion change from 0.9/0.1 to 0.5/0.5 is retained. |

The caption branch in `validate.py` currently uses **reference caption only**.
The separate cache and experiment scripts below use **reference caption + delta
caption**. Their results must not be assumed identical to `validate.py`.
`generate_test_submission.py` is unchanged and retains its original two-branch
0.5/0.5 test fusion.

## New scripts

| Script under `SEARLE_CIG/src/` | Purpose |
|---|---|
| `cache_cirr_query_embeddings.py` | Encode ref, gen and caption+delta queries once. Read `caption_extend["0"]` from existing `captions_ext` JSON. Supports validation and test1. |
| `evaluate_cirr_query_fusions.py` | Evaluate single branches and equal-weight branch combinations on validation. Encodes candidate images when run. |
| `evaluate_cached_cirr_three_branches.py` | Add caption+delta embeddings to existing `query_agreement_seed1600.npz` and `seed1601.npz` caches, reuse candidate embeddings, evaluate all branch combinations, save top-50 predictions. |
| `sweep_cirr_three_branch_weights.py` | Evaluate 68 unique normalized weight combinations on both validation seeds, including the five requested ratios and a simplex grid with step 0.1. Save JSON and Markdown results. |
| `retrieve_cached_cirr_test1.py` | Cache test1 candidate embeddings and export full top-50 and subset top-3 CIRR submissions for the 68 validation ratios. |
| `experiment_blip_cirr.py` | Separate BLIP-base experiment. Caption each distinct ref image once, encode caption+delta, reuse ref/gen/index embeddings, evaluate 68 ratios and export selected test1 submissions. Write everything under `analysis/blip/`; do not overwrite original code, captions or caches. |

## Required local data

Install dependencies from `requirements.txt` in a GPU environment. The scripts
use repository-relative paths; run commands from the repository root.

- `data/cirr/captions/cap.rc2.{val,test1}.json`
- `data/cirr/image_splits/split.rc2.{val,test1}.json` and images at the paths in these files
- `data/cirr/captions_ext/cap.ext.rc2.{val,test1}.json` for the existing caption branch
- Generated images named `{pairid:05d}.png`
- CLIP ViT-B/32 weights at `ViT-B-32.pt` for the cache reuse experiments

The existing JSON caption source is not verified to be BLIP. The BLIP experiment
uses `Salesforce/blip-image-captioning-base`, deterministic beam search with
3 beams and at most 32 new tokens.

Datasets, captions, model weights, generated images, embeddings and submissions
are excluded from Git. They must be supplied separately. Scripts that reuse
older caches need their named files under `analysis/`.

## Encode queries from images

```bash
python SEARLE_CIG/src/cache_cirr_query_embeddings.py \
  --dataset-path data \
  --generated-image-dir data/generated_images_test1_seed1600 \
  --cache-path analysis/cirr_three_branch_cache_test1_seed1600.pt \
  --split test --eval-type searle --batch-size 32 --num-workers 4
```

For validation, use `--split val` and the matching generated-image directory.

## Reuse existing validation caches and export test1

```bash
python SEARLE_CIG/src/evaluate_cached_cirr_three_branches.py
python SEARLE_CIG/src/sweep_cirr_three_branch_weights.py
python SEARLE_CIG/src/retrieve_cached_cirr_test1.py
```

The first command requires both `analysis/query_agreement_seed1600.npz` and
`analysis/query_agreement_seed1601.npz`, containing `q`, `qg`, `index`,
`index_names`, `pids`, `refs`, and `targets`. The sweep produces the ratios used
by test1 export. Test1 has no public ground-truth target for local Recall.

## Separate BLIP comparison

```bash
HF_HUB_ENABLE_HF_TRANSFER=0 python SEARLE_CIG/src/experiment_blip_cirr.py
```

Requires the validation and test1 caches from the preceding commands. Caption
files are checkpointed by image name so interrupted caption generation can
resume. The experiment compares BLIP and existing captions on the same cached
ref/gen/index branches. Baseline test1 predictions are unchanged when caption
weight is zero.

## Validation performed in this workspace

- 4,181 validation queries and 2,297 candidates, seeds 1600 and 1601.
- Existing captions: equal-thirds mean R@1 26.82%; 0.4/0.3/0.3 mean R@1 27.02%.
- BLIP-base captions: equal-thirds mean R@1 23.42%; 0.4/0.3/0.3 mean R@1 23.70%.
- Two-branch 0.5/0.5/0 mean R@1 26.45% for either caption source.
- 4,148 test1 queries and 2,315 candidates; 68 original-caption submission pairs
  and 9 separate BLIP submission pairs validated for sizes and unique predictions.
- Before and after BLIP, SHA-256 checks confirmed 24 original code/caption/cache
  files unchanged; baseline test1 full and subset predictions matched exactly.

These weights were selected on validation, not evaluated on an independent test
set. Full result files remain local under `analysis/`.
