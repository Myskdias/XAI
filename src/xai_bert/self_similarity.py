from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .embeddings import flatten_valid_tokens
from .modeling import BertHiddenStateRunner
from .types import PositionProbeDatasetTensors, FlattenedLayerData, BertBatchOutput
from .clipping import clip_embedding_dimensions


def mean_pairwise_cosine_similarity(embeddings: torch.Tensor) -> float:
    """
    Compute the mean cosine similarity between all pairs of embeddings.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings of shape (N, d), got {embeddings.shape}")

    norm = embeddings.norm(dim=1, keepdim=True)
    normalized = embeddings / (norm + 1e-8)

    sim_matrix = normalized @ normalized.T
    n = embeddings.shape[0]
    sum_sim = sim_matrix.sum() - n  # exclude self-similarity
    count_pairs = n * (n - 1)

    return (sum_sim / count_pairs).item()

def compute_word_self_similarity_single_layer(
    embeddings: torch.Tensor,
    dim_to_clip: int,
    anisotropy: Dict[str, float], # default to 0 meaning not active
) -> Dict[str, float]:
    """
    Compute mean pairwise cosine similarity before and after clipping selected dimensions.
    """
    before = mean_pairwise_cosine_similarity(embeddings)
    before -= anisotropy["before"]

    clipped_embeddings = clip_embedding_dimensions(embeddings, dims_to_clip=[dim_to_clip])

    after = mean_pairwise_cosine_similarity(clipped_embeddings)
    after -= anisotropy["after"]

    return {
        "before": before,
        "after": after,
        "difference": after - before,
    }

def compute_word_self_similarity_selected_layers(
    runner: BertHiddenStateRunner,
    texts: List[str],
    layer_indices: List[int],
    dims_to_clip: List[int],
    batch_size: int = 16,
    remove_special_tokens: bool = False,
    anisotropy: Dict[int, Dict[str, float]] = None, # default to None meaning not active
) -> Dict[int, Dict[str, float]]:
    """
    Compute self-similarity before and after clipping for selected layers.
    """
    results = {}
    batches_out = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_out = runner.run_batch(batch_texts)
        batches_out.append(batch_out)

    for layer_idx in layer_indices:
        embeddings = []
        for i in range(len(batches_out)):
            flat_layer_batch = flatten_valid_tokens(
                batch_output=batches_out[i],
                layer_idx=layer_idx,
                remove_special_tokens=remove_special_tokens,
            )
            embeddings.append(flat_layer_batch.embeddings)
        results[layer_idx] = compute_word_self_similarity_single_layer(
            embeddings=torch.cat(embeddings),
            dim_to_clip=dims_to_clip[layer_idx] if dims_to_clip else None,
            anisotropy=anisotropy[layer_idx],
        )
    return results