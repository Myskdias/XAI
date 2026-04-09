from collections import Counter
from typing import Dict, List, Tuple

import torch

from .embeddings import flatten_valid_tokens
from .modeling import BertHiddenStateRunner
from .types import FlattenedLayerData


def count_argmin_argmax_for_layer(
    flat_layer: FlattenedLayerData,
) -> Tuple[Counter, Counter]:
    """
    Count how often each dimension is the argmin/argmax across token embeddings.
    """
    embeddings = flat_layer.embeddings

    min_dims = torch.argmin(embeddings, dim=1).tolist()
    max_dims = torch.argmax(embeddings, dim=1).tolist()

    min_counter = Counter(min_dims)
    max_counter = Counter(max_dims)

    return min_counter, max_counter


def dominant_outlier_stats_for_layer(
    flat_layer: FlattenedLayerData,
) -> Dict[str, object]:
    """
    Return the most frequent argmin/argmax dimensions and their proportions.
    """
    min_counter, max_counter = count_argmin_argmax_for_layer(flat_layer)
    n_tokens = flat_layer.embeddings.shape[0]

    top_min_dim, top_min_count = min_counter.most_common(1)[0]
    top_max_dim, top_max_count = max_counter.most_common(1)[0]

    return {
        "n_tokens": n_tokens,
        "top_min_dim": top_min_dim,
        "top_min_count": top_min_count,
        "top_min_ratio": top_min_count / n_tokens,
        "top_max_dim": top_max_dim,
        "top_max_count": top_max_count,
        "top_max_ratio": top_max_count / n_tokens,
        "min_counter": min_counter,
        "max_counter": max_counter,
    }


def compute_outlier_dimension_stats_over_corpus(
    runner: BertHiddenStateRunner,
    texts: List[str],
    batch_size: int = 16,
    exclude_embedding_layer: bool = True,
    remove_special_tokens: bool = False,
) -> Dict[int, Dict[str, object]]:
    """
    For each layer, count how often each dimension is argmin/argmax
    across all valid tokens in the corpus.
    """
    start_layer = 1 if exclude_embedding_layer else 0
    n_layers = 13

    layer_min_counters = {}
    layer_max_counters = {}
    layer_token_counts = {}

    for layer_idx in range(start_layer, n_layers):
        layer_min_counters[layer_idx] = Counter()
        layer_max_counters[layer_idx] = Counter()
        layer_token_counts[layer_idx] = 0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_out = runner.run_batch(batch_texts)

        for layer_idx in range(start_layer, n_layers):
            flat = flatten_valid_tokens(
                batch_output=batch_out,
                layer_idx=layer_idx,
                remove_special_tokens=remove_special_tokens,
            )

            min_counter, max_counter = count_argmin_argmax_for_layer(flat)
            layer_min_counters[layer_idx].update(min_counter)
            layer_max_counters[layer_idx].update(max_counter)
            layer_token_counts[layer_idx] += flat.embeddings.shape[0]

    results = {}

    for layer_idx in range(start_layer, n_layers):
        n_tokens = layer_token_counts[layer_idx]
        min_counter = layer_min_counters[layer_idx]
        max_counter = layer_max_counters[layer_idx]

        top_min_dim, top_min_count = min_counter.most_common(1)[0]
        top_max_dim, top_max_count = max_counter.most_common(1)[0]

        results[layer_idx] = {
            "n_tokens": n_tokens,
            "top_min_dim": top_min_dim,
            "top_min_count": top_min_count,
            "top_min_ratio": top_min_count / n_tokens,
            "top_max_dim": top_max_dim,
            "top_max_count": top_max_count,
            "top_max_ratio": top_max_count / n_tokens,
            "min_counter": min_counter,
            "max_counter": max_counter,
        }

    return results


def print_outlier_stats(results: Dict[int, Dict[str, object]]) -> None:
    for layer_idx, stats in results.items():
        print(f"Layer {layer_idx}")
        print(f"  n_tokens      : {stats['n_tokens']}")
        print(
            f"  top min dim   : {stats['top_min_dim']} "
            f"({stats['top_min_ratio']:.2%})"
        )
        print(
            f"  top max dim   : {stats['top_max_dim']} "
            f"({stats['top_max_ratio']:.2%})"
        )
        print()


def top_k_dims(counter: Counter, k: int = 10):
    return counter.most_common(k)
