from typing import Dict, List

import torch

from .modeling import BertHiddenStateRunner
from .types import BertBatchOutput, FlattenedLayerData


def flatten_valid_tokens(
    batch_output: BertBatchOutput,
    layer_idx: int,
    remove_special_tokens: bool = False,
) -> FlattenedLayerData:
    """
    Flatten one layer from shape (B, T, d) to (N_valid_tokens, d),
    removing PAD tokens using attention_mask == 0.

    If remove_special_tokens=True, also removes [CLS] and [SEP].
    """
    layer_tensor = batch_output.hidden_states[layer_idx]
    attention_mask = batch_output.attention_mask
    tokens = batch_output.tokens

    valid_embeddings = []
    valid_positions = []
    valid_batch_indices = []
    valid_token_indices = []
    valid_token_strings = []

    batch_size, seq_len, _ = layer_tensor.shape

    for batch_idx in range(batch_size):
        for token_idx in range(seq_len):
            if attention_mask[batch_idx, token_idx].item() == 0:
                continue

            tok = tokens[batch_idx][token_idx]
            if remove_special_tokens and tok in ("[CLS]", "[SEP]"):
                continue

            valid_embeddings.append(layer_tensor[batch_idx, token_idx])
            valid_positions.append(token_idx)
            valid_batch_indices.append(batch_idx)
            valid_token_indices.append(token_idx)
            valid_token_strings.append(tok)

    if not valid_embeddings:
        raise ValueError("No valid tokens found after filtering.")

    return FlattenedLayerData(
        embeddings=torch.stack(valid_embeddings, dim=0),
        positions=torch.tensor(valid_positions, dtype=torch.long),
        batch_indices=torch.tensor(valid_batch_indices, dtype=torch.long),
        token_indices=torch.tensor(valid_token_indices, dtype=torch.long),
        token_strings=valid_token_strings,
    )


def mean_vector_for_layer(
    batch_output: BertBatchOutput,
    layer_idx: int,
    remove_special_tokens: bool = False,
) -> torch.Tensor:
    """
    Compute the mean vector of one layer over valid tokens.
    Returns shape (d,).
    """
    flat = flatten_valid_tokens(
        batch_output=batch_output,
        layer_idx=layer_idx,
        remove_special_tokens=remove_special_tokens,
    )
    return flat.embeddings.mean(dim=0)


def mean_vectors_all_transformer_layers(
    batch_output: BertBatchOutput,
    remove_special_tokens: bool = False,
    exclude_embedding_layer: bool = True,
) -> Dict[int, torch.Tensor]:
    """
    Compute mean vector per layer.
    By default excludes hidden_states[0] (embedding layer).
    """
    start = 1 if exclude_embedding_layer else 0
    result = {}

    for layer_idx in range(start, len(batch_output.hidden_states)):
        result[layer_idx] = mean_vector_for_layer(
            batch_output=batch_output,
            layer_idx=layer_idx,
            remove_special_tokens=remove_special_tokens,
        )

    return result


def compute_mean_vectors_over_corpus(
    runner: BertHiddenStateRunner,
    texts: List[str],
    batch_size: int = 16,
    exclude_embedding_layer: bool = True,
    remove_special_tokens: bool = False,
) -> Dict[int, torch.Tensor]:
    """
    Returns mean vector per layer over the whole corpus.
    """
    start_layer = 1 if exclude_embedding_layer else 0
    n_layers = 13

    layer_sums = {}
    layer_counts = {}

    for layer_idx in range(start_layer, n_layers):
        layer_sums[layer_idx] = None
        layer_counts[layer_idx] = 0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_out = runner.run_batch(batch_texts)

        for layer_idx in range(start_layer, n_layers):
            flat = flatten_valid_tokens(
                batch_output=batch_out,
                layer_idx=layer_idx,
                remove_special_tokens=remove_special_tokens,
            )

            batch_sum = flat.embeddings.sum(dim=0)
            batch_count = flat.embeddings.shape[0]

            if layer_sums[layer_idx] is None:
                layer_sums[layer_idx] = batch_sum
            else:
                layer_sums[layer_idx] += batch_sum

            layer_counts[layer_idx] += batch_count

    mean_vectors = {}
    for layer_idx in range(start_layer, n_layers):
        if layer_counts[layer_idx] == 0:
            raise ValueError(f"No valid tokens for layer {layer_idx}")
        mean_vectors[layer_idx] = layer_sums[layer_idx] / layer_counts[layer_idx]

    return mean_vectors


def collect_layer_embeddings_over_corpus(
    runner: BertHiddenStateRunner,
    texts: List[str],
    layer_idx: int,
    batch_size: int = 16,
    remove_special_tokens: bool = False,
) -> torch.Tensor:
    """
    Collect all valid token embeddings for one layer over a corpus.
    Returns a tensor of shape (N, d).
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_out = runner.run_batch(batch_texts)

        flat = flatten_valid_tokens(
            batch_output=batch_out,
            layer_idx=layer_idx,
            remove_special_tokens=remove_special_tokens,
        )

        all_embeddings.append(flat.embeddings)

    if not all_embeddings:
        raise ValueError("No embeddings collected from corpus.")

    return torch.cat(all_embeddings, dim=0)
