from .anisotropy import (
    anisotropy_all_layers_before_after,
    anisotropy_before_after_clipping,
    mean_random_pairwise_cosine,
    print_anisotropy_results,
)
from .clipping import (
    clip_embedding_dimensions,
    clip_flattened_layer_data,
    compare_outlier_stats_before_after_clipping,
    print_before_after_comparison,
    select_primary_outlier_dims,
)
from .data import (
    load_sst2_text_splits,
    load_wic_splits,
    context_words_for_text,
)
from .embeddings import (
    collect_layer_embeddings_over_corpus,
    compute_mean_vectors_over_corpus,
    flatten_valid_tokens,
    mean_vector_for_layer,
    mean_vectors_all_transformer_layers,
)
from .modeling import BertHiddenStateRunner
from .outliers import (
    compute_outlier_dimension_stats_over_corpus,
    count_argmin_argmax_for_layer,
    dominant_outlier_stats_for_layer,
    print_outlier_stats,
    top_k_dims,
)
from .plotting import plot_anisotropy_before_after, plot_mean_vectors
from .probe import (
    LinearPositionProbe,
    PositionProbeDataset,
    build_position_probe_tensors,
    contribution_vectors_for_dataset,
    evaluate_probe,
    mean_contribution_of_dim_by_position,
    run_probe_for_selected_layers,
    run_probe_for_selected_layers_with_splits,
    run_single_layer_position_probe,
    run_single_layer_position_probe_with_splits,
    split_probe_tensors,
    split_texts,
    train_position_probe,
)
from .types import BertBatchOutput, FlattenedLayerData, PositionProbeDatasetTensors

from .self_similarity import (
    compute_word_self_similarity_selected_layers,
)
from .token_outlier import (
    detect_token_outlier_responsibility,
    print_token_outlier_summary,
)
from .word_sense import (
    contextual_word_vector,
    evaluate_wic_with_threshold,
)

__all__ = [
    "BertBatchOutput",
    "FlattenedLayerData",
    "PositionProbeDatasetTensors",
    "BertHiddenStateRunner",
    "flatten_valid_tokens",
    "mean_vector_for_layer",
    "mean_vectors_all_transformer_layers",
    "compute_mean_vectors_over_corpus",
    "collect_layer_embeddings_over_corpus",
    "count_argmin_argmax_for_layer",
    "dominant_outlier_stats_for_layer",
    "compute_outlier_dimension_stats_over_corpus",
    "print_outlier_stats",
    "top_k_dims",
    "clip_embedding_dimensions",
    "clip_flattened_layer_data",
    "select_primary_outlier_dims",
    "compare_outlier_stats_before_after_clipping",
    "print_before_after_comparison",
    "load_sst2_text_splits",
    "load_wic_splits",
    "context_words_for_text",
    "mean_random_pairwise_cosine",
    "anisotropy_before_after_clipping",
    "anisotropy_all_layers_before_after",
    "print_anisotropy_results",
    "plot_mean_vectors",
    "plot_anisotropy_before_after",
    "PositionProbeDataset",
    "LinearPositionProbe",
    "build_position_probe_tensors",
    "split_probe_tensors",
    "evaluate_probe",
    "train_position_probe",
    "contribution_vectors_for_dataset",
    "mean_contribution_of_dim_by_position",
    "split_texts",
    "run_single_layer_position_probe",
    "run_single_layer_position_probe_with_splits",
    "run_probe_for_selected_layers",
    "run_probe_for_selected_layers_with_splits",
    "compute_word_self_similarity_selected_layers",
    "detect_token_outlier_responsibility",
    "print_token_outlier_summary",
    "contextual_word_vector",
    "evaluate_wic_with_threshold",
]
