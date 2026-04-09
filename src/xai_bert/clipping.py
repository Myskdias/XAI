from typing import Dict, Iterable, List, Union

import torch

from .outliers import dominant_outlier_stats_for_layer
from .types import FlattenedLayerData


def clip_embedding_dimensions(
    embeddings: torch.Tensor,
    dims_to_clip: Union[int, Iterable[int]],
) -> torch.Tensor:
    """
    Return a clipped copy of embeddings by zeroing selected dimensions.
    """
    if isinstance(dims_to_clip, int):
        dims_to_clip = [dims_to_clip]
    else:
        dims_to_clip = list(dims_to_clip)

    clipped = embeddings.clone()
    clipped[..., dims_to_clip] = 0.0
    return clipped


def clip_flattened_layer_data(
    flat_layer: FlattenedLayerData,
    dims_to_clip: Union[int, Iterable[int]],
) -> FlattenedLayerData:
    """
    Return a new FlattenedLayerData with clipped embeddings.
    """
    clipped_embeddings = clip_embedding_dimensions(
        flat_layer.embeddings,
        dims_to_clip=dims_to_clip,
    )

    return FlattenedLayerData(
        embeddings=clipped_embeddings,
        positions=flat_layer.positions.clone(),
        batch_indices=flat_layer.batch_indices.clone(),
        token_indices=flat_layer.token_indices.clone(),
        token_strings=list(flat_layer.token_strings),
    )


def select_primary_outlier_dims(
    layer_stats: Dict[str, object],
    mode: str = "min",
) -> List[int]:
    """
    Select primary outlier dimensions from one layer stats dict.

    mode:
    - "min": take top_min_dim
    - "max": take top_max_dim
    - "both": take both
    """
    if mode == "min":
        return [int(layer_stats["top_min_dim"])]
    if mode == "max":
        return [int(layer_stats["top_max_dim"])]
    if mode == "both":
        dims = [int(layer_stats["top_min_dim"]), int(layer_stats["top_max_dim"])]
        return sorted(set(dims))

    raise ValueError(f"Unknown mode: {mode}")


def compare_outlier_stats_before_after_clipping(
    flat_layer: FlattenedLayerData,
    dims_to_clip: Union[int, Iterable[int]],
) -> Dict[str, object]:
    """
    Compare dominant argmin/argmax stats before and after clipping.
    """
    before = dominant_outlier_stats_for_layer(flat_layer)
    clipped_flat = clip_flattened_layer_data(flat_layer, dims_to_clip=dims_to_clip)
    after = dominant_outlier_stats_for_layer(clipped_flat)

    return {
        "before": before,
        "after": after,
        "dims_clipped": list([dims_to_clip] if isinstance(dims_to_clip, int) else dims_to_clip),
    }


def print_before_after_comparison(comparison: Dict[str, object]) -> None:
    before = comparison["before"]
    after = comparison["after"]
    dims_clipped = comparison["dims_clipped"]

    print(f"Clipped dims: {dims_clipped}")
    print("Before clipping:")
    print(
        f"  top min dim: {before['top_min_dim']} ({before['top_min_ratio']:.2%}), "
        f"top max dim: {before['top_max_dim']} ({before['top_max_ratio']:.2%})"
    )
    print("After clipping:")
    print(
        f"  top min dim: {after['top_min_dim']} ({after['top_min_ratio']:.2%}), "
        f"top max dim: {after['top_max_dim']} ({after['top_max_ratio']:.2%})"
    )
