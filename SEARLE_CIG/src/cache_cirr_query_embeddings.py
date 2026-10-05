"""Cache the three CIRR query branches used for fusion experiments.

The cache contains one normalized CLIP-space vector per pair for:
  * ``ref``: reference-image pseudo-token + delta caption;
  * ``gen``: diffusion-image pseudo-token + delta caption;
  * ``text``: reference-image caption + delta caption.

``captions_ext/cap.ext.rc2.<split>.json`` supplies the reference caption
(``caption_extend["0"]``).  This avoids generating a new caption for every
experiment and keeps all fusion runs aligned to exactly the same queries.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import PROJECT_ROOT, collate_fn, targetpad_transform
from datasets import CIRRDataset
from encode_with_pseudo_tokens import encode_with_pseudo_tokens
from phi import Phi
from utils_feat import device


CACHE_SCHEMA_VERSION = 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reference_captions(dataset_path: Path, split: str) -> Tuple[Dict[int, str], Path]:
    captions_path = dataset_path / "cirr" / "captions_ext" / f"cap.ext.rc2.{split}.json"
    if not captions_path.is_file():
        raise FileNotFoundError(
            f"Reference-caption file not found: {captions_path}. "
            "Expected captions_ext/cap.ext.rc2.<split>.json."
        )

    with captions_path.open() as file:
        records = json.load(file)

    captions = {}
    for record in records:
        caption = record.get("caption_extend", {}).get("0", "").strip()
        # The extension annotations use bracketed markers for missing content.
        if caption.startswith("["):
            caption = ""
        captions[int(record["pairid"])] = caption
    return captions, captions_path


def compose_text_prompt(reference_caption: str, delta_caption: str) -> str:
    reference_caption = reference_caption.strip(" .")
    delta_caption = delta_caption.strip(" .")
    if reference_caption:
        return f"a photo of {reference_caption}. {delta_caption}"
    return f"a photo that {delta_caption}"


def load_phi_and_clip(args: argparse.Namespace):
    if args.eval_type == "phi":
        if args.exp_name is None:
            raise ValueError("--exp-name is required with --eval-type phi")
        phi_path = PROJECT_ROOT / "data" / "phi_models" / args.exp_name
        hyperparameters_path = phi_path / "hyperparameters.json"
        if not hyperparameters_path.is_file():
            raise FileNotFoundError(f"Phi experiment not found: {phi_path}")
        if args.phi_checkpoint_name is None:
            raise ValueError("--phi-checkpoint-name is required with --eval-type phi")

        with hyperparameters_path.open() as file:
            hyperparameters = json.load(file)
        clip_model_name = hyperparameters["clip_model_name"]
        clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False, download_root="./")
        phi = Phi(
            input_dim=clip_model.visual.output_dim,
            hidden_dim=clip_model.visual.output_dim * 4,
            output_dim=clip_model.token_embedding.embedding_dim,
            dropout=hyperparameters["phi_dropout"],
        ).to(device)
        checkpoint = torch.load(phi_path / "checkpoints" / args.phi_checkpoint_name, map_location=device)
        phi.load_state_dict(checkpoint[phi.__class__.__name__])
        phi = phi.eval()
    else:
        clip_model_name = "ViT-B/32" if args.eval_type == "searle" else "ViT-L/14"
        phi, _ = torch.hub.load("miccunifi/SEARLE", "searle", source="github", backbone=clip_model_name)
        phi = phi.to(device).eval()
        clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False, download_root="./")

    clip_model = clip_model.float().eval().requires_grad_(False).to(device)
    if args.preprocess_type == "targetpad":
        preprocess = targetpad_transform(1.25, clip_model.visual.input_resolution)
    else:
        preprocess = clip_preprocess
    return phi, clip_model, clip_model_name, preprocess


@torch.no_grad()
def create_cache(args: argparse.Namespace) -> dict:
    dataset_path = Path(args.dataset_path)
    reference_captions, reference_caption_path = load_reference_captions(dataset_path, args.split)
    phi, clip_model, clip_model_name, preprocess = load_phi_and_clip(args)
    dataset = CIRRDataset(
        dataset_path, args.split, "relative", preprocess,
        generated_image_dir=args.generated_image_dir,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=False, collate_fn=collate_fn, shuffle=False,
    )

    ref_queries: List[torch.Tensor] = []
    gen_queries: List[torch.Tensor] = []
    text_queries: List[torch.Tensor] = []
    pair_ids: List[int] = []
    text_prompts: List[str] = []

    for batch in tqdm(loader, desc="Encoding CIRR query branches"):
        ids = [int(pair_id) for pair_id in batch["pair_id"]]
        missing = [pair_id for pair_id in ids if pair_id not in reference_captions]
        if missing:
            raise KeyError(f"Missing reference caption for pair ids: {missing[:5]}")

        delta_captions = list(batch["relative_caption"])
        pseudo_prompt = [f"a photo of $ that {caption}" for caption in delta_captions]
        text_prompt = [
            compose_text_prompt(reference_captions[pair_id], delta_caption)
            for pair_id, delta_caption in zip(ids, delta_captions)
        ]

        ref_features = clip_model.encode_image(batch["reference_image"].to(device))
        gen_features = clip_model.encode_image(batch["generated_image"].to(device))
        ref_tokens = phi(ref_features)
        gen_tokens = phi(gen_features)
        pseudo_tokens = clip.tokenize(pseudo_prompt, context_length=77, truncate=True).to(device)
        text_tokens = clip.tokenize(text_prompt, context_length=77, truncate=True).to(device)

        ref_queries.append(F.normalize(encode_with_pseudo_tokens(clip_model, pseudo_tokens, ref_tokens), dim=-1).cpu())
        gen_queries.append(F.normalize(encode_with_pseudo_tokens(clip_model, pseudo_tokens, gen_tokens), dim=-1).cpu())
        text_queries.append(F.normalize(clip_model.encode_text(text_tokens), dim=-1).cpu())
        pair_ids.extend(ids)
        text_prompts.extend(text_prompt)

    cache = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "metadata": {
            "dataset": "cirr",
            "split": args.split,
            "clip_model_name": clip_model_name,
            "eval_type": args.eval_type,
            "preprocess_type": args.preprocess_type,
            "reference_caption_file": str(reference_caption_path),
            "reference_caption_sha256": file_sha256(reference_caption_path),
        },
        "pair_ids": pair_ids,
        "ref": torch.vstack(ref_queries),
        "gen": torch.vstack(gen_queries),
        "text": torch.vstack(text_queries),
        "text_prompts": text_prompts,
    }
    return cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache CIRR ref/gen/text query embeddings.")
    parser.add_argument("--dataset-path", required=True, help="Dataset root containing cirr/")
    parser.add_argument("--generated-image-dir", required=True)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--eval-type", choices=["phi", "searle", "searle-xl"], default="searle")
    parser.add_argument("--exp-name", help="Phi experiment name; required for --eval-type phi")
    parser.add_argument("--phi-checkpoint-name", help="Phi checkpoint; required for --eval-type phi")
    parser.add_argument("--preprocess-type", choices=["clip", "targetpad"], default="targetpad")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing cache.")
    args = parser.parse_args()
    args.split = "test1" if args.split == "test" else args.split

    cache_path = Path(args.cache_path)
    if cache_path.exists() and not args.force:
        raise FileExistsError(f"Cache already exists: {cache_path}. Use --force to replace it.")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = create_cache(args)
    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(cache, temporary_path)
    temporary_path.replace(cache_path)
    print(f"Saved {len(cache['pair_ids'])} query triplets to {cache_path}")


if __name__ == "__main__":
    main()
