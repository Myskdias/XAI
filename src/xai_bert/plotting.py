from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch


def plot_mean_vectors(
    mean_vectors: Dict[int, torch.Tensor],
    title: str = "Average vectors for each layer of BERT-base",
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    plt.figure(figsize=figsize)

    for layer_idx, mean_vec in mean_vectors.items():
        x = torch.arange(mean_vec.shape[0]).numpy()
        y = mean_vec.numpy()
        plt.plot(x, y, linewidth=1, label=f"Layer {layer_idx}")

    plt.xlabel("Dimension index")
    plt.ylabel("Average activation")
    plt.title(title)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_anisotropy_before_after(
    results: Dict[int, Dict[str, object]],
    title: str = "Anisotropy before and after clipping",
) -> None:
    layers = sorted(results.keys())
    before = [results[layer]["before"] for layer in layers]
    after = [results[layer]["after"] for layer in layers]

    plt.figure(figsize=(8, 5))
    plt.plot(layers, before, marker="o", label="Before clipping")
    plt.plot(layers, after, marker="o", label="After clipping")
    plt.xlabel("Layer")
    plt.ylabel("Mean cosine similarity")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
