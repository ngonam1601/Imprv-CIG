"""Evaluate single-branch and fusion ablations from a CIRR query-embedding cache."""

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List

import clip
import numpy as np
import torch
import torch.nn.functional as F

from data_utils import targetpad_transform
from datasets import CIRRDataset
from utils_feat import device, extract_image_features


BRANCHES = ("ref", "gen", "text")


def fusion_names() -> Iterable[str]:
    for count in range(1, len(BRANCHES) + 1):
        yield from ("+".join(branches) for branches in combinations(BRANCHES, count))


def fuse(cache: dict, name: str) -> torch.Tensor:
    branches = name.split("+")
    invalid = set(branches).difference(BRANCHES)
    if invalid:
        raise ValueError(f"Unknown fusion branch(es): {sorted(invalid)}")
    return F.normalize(sum(F.normalize(cache[branch].float(), dim=-1) for branch in branches), dim=-1)


@torch.no_grad()
def evaluate_fusion(
    query_features: torch.Tensor,
    index_features: torch.Tensor,
    index_names: List[str],
    triplets: List[dict],
) -> Dict[str, float]:
    distances = 1 - query_features.to(device) @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu().numpy()
    sorted_names = np.asarray(index_names)[sorted_indices]

    recalls = {1: 0, 5: 0, 10: 0, 50: 0}
    group_recalls = {1: 0, 2: 0, 3: 0}
    for row, triplet in zip(sorted_names, triplets):
        row = row[row != triplet["reference"]]
        target = triplet["target_hard"]
        target_rank = int(np.flatnonzero(row == target)[0])
        for k in recalls:
            recalls[k] += target_rank < k

        group = set(triplet["img_set"]["members"])
        group_row = [name for name in row if name in group]
        group_rank = group_row.index(target)
        for k in group_recalls:
            group_recalls[k] += group_rank < k

    total = len(triplets)
    return {
        **{f"recall_at{k}": 100 * value / total for k, value in recalls.items()},
        **{f"group_recall_at{k}": 100 * value / total for k, value in group_recalls.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CIRR query branch fusion ablations.")
    parser.add_argument("--dataset-path", required=True, help="Dataset root containing cirr/")
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--fusions", nargs="+", default=list(fusion_names()),
        help="Space-separated branch combinations, e.g. ref gen text ref+gen ref+gen+text",
    )
    args = parser.parse_args()

    cache = torch.load(args.cache_path, map_location="cpu")
    metadata = cache.get("metadata", {})
    if cache.get("schema_version") != 1 or metadata.get("dataset") != "cirr":
        raise ValueError("The cache is not a supported CIRR query-embedding cache.")
    if metadata.get("split") != "val":
        raise ValueError("Recall evaluation requires a validation cache (split=val).")
    if any(branch not in cache for branch in BRANCHES):
        raise ValueError("Cache must contain ref, gen, and text embeddings.")

    dataset_path = Path(args.dataset_path)
    captions_path = dataset_path / "cirr" / "captions" / "cap.rc2.val.json"
    with captions_path.open() as file:
        triplets = json.load(file)
    cached_pair_ids = [int(pair_id) for pair_id in cache["pair_ids"]]
    pair_ids = [int(triplet["pairid"]) for triplet in triplets]
    if cached_pair_ids != pair_ids:
        raise ValueError("Cache pair order does not match CIRR validation annotations.")

    clip_model_name = metadata["clip_model_name"]
    clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False, download_root="./")
    clip_model = clip_model.float().eval().requires_grad_(False)
    preprocess = (
        targetpad_transform(1.25, clip_model.visual.input_resolution)
        if metadata.get("preprocess_type") == "targetpad" else clip_preprocess
    )
    index_dataset = CIRRDataset(dataset_path, "val", "classic", preprocess)
    index_features, index_names = extract_image_features(index_dataset, clip_model)
    index_features = F.normalize(index_features.to(device).float(), dim=-1)

    results = {
        "cache_path": str(Path(args.cache_path)),
        "cache_metadata": metadata,
        "fusions": {},
    }
    for name in args.fusions:
        results["fusions"][name] = evaluate_fusion(fuse(cache, name), index_features, index_names, triplets)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        json.dump(results, file, indent=2)
    for name, metrics in results["fusions"].items():
        print(name, " ".join(f"{metric}={value:.2f}" for metric, value in metrics.items()))


if __name__ == "__main__":
    main()
