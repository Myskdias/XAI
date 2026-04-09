from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .clipping import clip_embedding_dimensions, clip_flattened_layer_data, select_primary_outlier_dims
from .embeddings import collect_layer_embeddings_over_corpus
from .modeling import BertHiddenStateRunner
from .types import FlattenedLayerData


def mean_random_pairwise_cosine(
    embeddings: torch.Tensor,
    n_pairs: int = 1000,
    seed: Optional[int] = 42,
) -> float:
    """
    Estimate anisotropy as the mean cosine similarity between random pairs of embeddings.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings of shape (N, d), got {embeddings.shape}")

    n_tokens = embeddings.shape[0]
    if n_tokens < 2:
        raise ValueError("Need at least 2 embeddings to compute pairwise cosine similarity.")

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    idx1 = torch.randint(0, n_tokens, (n_pairs,), generator=generator)
    idx2 = torch.randint(0, n_tokens, (n_pairs,), generator=generator)

    same = idx1 == idx2
    while same.any():
        idx2[same] = torch.randint(0, n_tokens, (same.sum().item(),), generator=generator)
        same = idx1 == idx2

    x1 = embeddings[idx1]
    x2 = embeddings[idx2]

    cos = F.cosine_similarity(x1, x2, dim=1)
    return cos.mean().item()


def anisotropy_before_after_clipping(
    flat_layer: FlattenedLayerData,
    dims_to_clip,
    n_pairs: int = 1000,
    seed: Optional[int] = 42,
) -> Dict[str, float]:
    """
    Compare anisotropy before and after clipping selected dimensions.
    """
    before_score = mean_random_pairwise_cosine(
        flat_layer.embeddings,
        n_pairs=n_pairs,
        seed=seed,
    )

    clipped_flat = clip_flattened_layer_data(
        flat_layer=flat_layer,
        dims_to_clip=dims_to_clip,
    )

    after_score = mean_random_pairwise_cosine(
        clipped_flat.embeddings,
        n_pairs=n_pairs,
        seed=seed,
    )

    return {
        "before": before_score,
        "after": after_score,
        "delta": after_score - before_score,
    }


def anisotropy_all_layers_before_after(
    runner: BertHiddenStateRunner,
    texts: List[str],
    outlier_stats: Dict[int, Dict[str, object]],
    batch_size: int = 16,
    remove_special_tokens: bool = False,
    exclude_embedding_layer: bool = True,
    clip_mode: str = "min",
    n_pairs: int = 1000,
    seed: Optional[int] = 42,
) -> Dict[int, Dict[str, object]]:
    """
    Compute anisotropy before/after clipping for all layers.
    """
    start_layer = 1 if exclude_embedding_layer else 0
    n_layers = 13

    results = {}

    for layer_idx in range(start_layer, n_layers):
        layer_embeddings = collect_layer_embeddings_over_corpus(
            runner=runner,
            texts=texts,
            layer_idx=layer_idx,
            batch_size=batch_size,
            remove_special_tokens=remove_special_tokens,
        )

        dims_to_clip = select_primary_outlier_dims(
            outlier_stats[layer_idx],
            mode=clip_mode,
        )

        before_score = mean_random_pairwise_cosine(
            layer_embeddings,
            n_pairs=n_pairs,
            seed=seed,
        )

        clipped_embeddings = clip_embedding_dimensions(
            layer_embeddings,
            dims_to_clip=dims_to_clip,
        )

        after_score = mean_random_pairwise_cosine(
            clipped_embeddings,
            n_pairs=n_pairs,
            seed=seed,
        )

        results[layer_idx] = {
            "before": before_score,
            "after": after_score,
            "delta": after_score - before_score,
            "dims_clipped": dims_to_clip,
        }

    return results


def print_anisotropy_results(results: Dict[int, Dict[str, object]]) -> None:
    for layer_idx, res in results.items():
        print(f"Layer {layer_idx}")
        print(f"  clipped dims : {res['dims_clipped']}")
        print(f"  before       : {res['before']:.6f}")
        print(f"  after        : {res['after']:.6f}")
        print(f"  delta        : {res['delta']:.6f}")
        print()
