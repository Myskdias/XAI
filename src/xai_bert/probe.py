from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .embeddings import flatten_valid_tokens
from .modeling import BertHiddenStateRunner
from .types import PositionProbeDatasetTensors


def build_position_probe_tensors(
    runner: BertHiddenStateRunner,
    texts: List[str],
    layer_idx: int,
    batch_size: int = 16,
    remove_special_tokens: bool = False,
    max_position: int = 300,
) -> PositionProbeDatasetTensors:
    """
    Build a token-level dataset for position prediction from one layer.

    Keeps only tokens whose position < max_position.
    """
    all_x = []
    all_y = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_out = runner.run_batch(batch_texts)

        flat = flatten_valid_tokens(
            batch_output=batch_out,
            layer_idx=layer_idx,
            remove_special_tokens=remove_special_tokens,
        )

        keep = flat.positions < max_position

        if keep.any():
            all_x.append(flat.embeddings[keep])
            all_y.append(flat.positions[keep])

    if not all_x:
        raise ValueError("No examples kept for position probe.")

    x = torch.cat(all_x, dim=0)
    y = torch.cat(all_y, dim=0)

    return PositionProbeDatasetTensors(X=x, y=y)


class PositionProbeDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x.float()
        self.y = y.long()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class LinearPositionProbe(nn.Module):
    def __init__(self, input_dim: int = 768, num_classes: int = 300):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes, bias=False)

    def forward(self, x):
        return self.linear(x)


def split_probe_tensors(
    x: torch.Tensor,
    y: torch.Tensor,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    n_examples = x.shape[0]
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_examples, generator=generator)

    x = x[perm]
    y = y[perm]

    n_train = int(train_ratio * n_examples)
    n_val = int(val_ratio * n_examples)

    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train : n_train + n_val], y[n_train : n_train + n_val]
    x_test, y_test = x[n_train + n_val :], y[n_train + n_val :]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def evaluate_probe(model, dataloader, device: str = "cpu"):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.numel()
            loss_sum += loss.item() * yb.size(0)

    return {
        "loss": loss_sum / total,
        "acc": correct / total,
    }


def train_position_probe(
    train_dataset: PositionProbeDataset,
    val_dataset: PositionProbeDataset,
    input_dim: int = 768,
    num_classes: int = 300,
    batch_size: int = 128,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: Optional[str] = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = LinearPositionProbe(input_dim=input_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            total_correct += (preds == yb).sum().item()
            total_examples += yb.numel()
            total_loss += loss.item() * yb.size(0)

        train_metrics = {
            "loss": total_loss / total_examples,
            "acc": total_correct / total_examples,
        }
        val_metrics = evaluate_probe(model, val_loader, device=device)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
            }
        )

    return model, history


def contribution_vectors_for_dataset(
    probe_model: LinearPositionProbe,
    x: torch.Tensor,
    y: torch.Tensor,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Compute contribution vectors c(i)=|w_i * v_i| for each example
    using the row of W corresponding to the true class y.
    """
    device = device or next(probe_model.parameters()).device
    weights = probe_model.linear.weight.detach().to(device)

    x = x.to(device)
    y = y.to(device)

    weights_y = weights[y]
    contributions = torch.abs(weights_y * x)
    return contributions.detach().cpu()


def mean_contribution_of_dim_by_position(
    probe_model: LinearPositionProbe,
    x: torch.Tensor,
    y: torch.Tensor,
    dim_idx: int,
    num_classes: int = 300,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    For one neuron dimension dim_idx, compute mean contribution by target position.
    Returns shape (num_classes,).
    """
    contributions = contribution_vectors_for_dataset(
        probe_model=probe_model,
        x=x,
        y=y,
        device=device,
    )

    contrib_dim = contributions[:, dim_idx]
    out = torch.zeros(num_classes)

    for pos in range(num_classes):
        mask = y == pos
        if mask.any():
            out[pos] = contrib_dim[mask].mean()

    return out


def split_texts(
    texts: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    n_texts = len(texts)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_texts, generator=generator).tolist()

    texts_shuffled = [texts[i] for i in perm]

    n_train = int(train_ratio * n_texts)
    n_val = int(val_ratio * n_texts)

    train_texts = texts_shuffled[:n_train]
    val_texts = texts_shuffled[n_train : n_train + n_val]
    test_texts = texts_shuffled[n_train + n_val :]

    return train_texts, val_texts, test_texts


def run_single_layer_position_probe(
    runner: BertHiddenStateRunner,
    texts: List[str],
    layer_idx: int,
    max_position: int = 300,
    build_batch_size: int = 16,
    probe_batch_size: int = 128,
    num_epochs: int = 10,
    lr: float = 1e-3,
    remove_special_tokens: bool = False,
    seed: int = 42,
    device: Optional[str] = None,
):
    train_texts, val_texts, test_texts = split_texts(texts, seed=seed)

    train_tensors = build_position_probe_tensors(
        runner=runner,
        texts=train_texts,
        layer_idx=layer_idx,
        batch_size=build_batch_size,
        remove_special_tokens=remove_special_tokens,
        max_position=max_position,
    )

    val_tensors = build_position_probe_tensors(
        runner=runner,
        texts=val_texts,
        layer_idx=layer_idx,
        batch_size=build_batch_size,
        remove_special_tokens=remove_special_tokens,
        max_position=max_position,
    )

    test_tensors = build_position_probe_tensors(
        runner=runner,
        texts=test_texts,
        layer_idx=layer_idx,
        batch_size=build_batch_size,
        remove_special_tokens=remove_special_tokens,
        max_position=max_position,
    )

    x_train, y_train = train_tensors.X, train_tensors.y
    x_val, y_val = val_tensors.X, val_tensors.y
    x_test, y_test = test_tensors.X, test_tensors.y

    train_ds = PositionProbeDataset(x_train, y_train)
    val_ds = PositionProbeDataset(x_val, y_val)
    test_ds = PositionProbeDataset(x_test, y_test)

    model, history = train_position_probe(
        train_dataset=train_ds,
        val_dataset=val_ds,
        input_dim=x_train.shape[1],
        num_classes=max_position,
        batch_size=probe_batch_size,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
    )

    test_loader = DataLoader(test_ds, batch_size=probe_batch_size, shuffle=False)
    test_metrics = evaluate_probe(
        model,
        test_loader,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )

    return {
        "model": model,
        "history": history,
        "test_metrics": test_metrics,
        "X_test": x_test,
        "y_test": y_test,
    }


def run_single_layer_position_probe_with_splits(
    runner: BertHiddenStateRunner,
    train_texts: List[str],
    val_texts: List[str],
    test_texts: List[str],
    layer_idx: int,
    max_position: int = 300,
    build_batch_size: int = 16,
    probe_batch_size: int = 128,
    num_epochs: int = 10,
    lr: float = 1e-3,
    remove_special_tokens: bool = False,
    device: Optional[str] = None,
):
    """
    Train and evaluate a position probe for a single layer using explicit
    train/validation/test text splits.
    """
    train_tensors = build_position_probe_tensors(
        runner=runner,
        texts=train_texts,
        layer_idx=layer_idx,
        batch_size=build_batch_size,
        remove_special_tokens=remove_special_tokens,
        max_position=max_position,
    )

    val_tensors = build_position_probe_tensors(
        runner=runner,
        texts=val_texts,
        layer_idx=layer_idx,
        batch_size=build_batch_size,
        remove_special_tokens=remove_special_tokens,
        max_position=max_position,
    )

    test_tensors = build_position_probe_tensors(
        runner=runner,
        texts=test_texts,
        layer_idx=layer_idx,
        batch_size=build_batch_size,
        remove_special_tokens=remove_special_tokens,
        max_position=max_position,
    )

    x_train, y_train = train_tensors.X, train_tensors.y
    x_val, y_val = val_tensors.X, val_tensors.y
    x_test, y_test = test_tensors.X, test_tensors.y

    train_ds = PositionProbeDataset(x_train, y_train)
    val_ds = PositionProbeDataset(x_val, y_val)
    test_ds = PositionProbeDataset(x_test, y_test)

    model, history = train_position_probe(
        train_dataset=train_ds,
        val_dataset=val_ds,
        input_dim=x_train.shape[1],
        num_classes=max_position,
        batch_size=probe_batch_size,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
    )

    test_loader = DataLoader(test_ds, batch_size=probe_batch_size, shuffle=False)
    test_metrics = evaluate_probe(
        model,
        test_loader,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )

    return {
        "model": model,
        "history": history,
        "test_metrics": test_metrics,
        "X_test": x_test,
        "y_test": y_test,
    }


def run_probe_for_selected_layers(
    runner: BertHiddenStateRunner,
    texts: List[str],
    layers: Tuple[int, ...] = (1, 6, 12),
    max_position: int = 300,
    build_batch_size: int = 16,
    probe_batch_size: int = 128,
    num_epochs: int = 10,
    lr: float = 1e-3,
    remove_special_tokens: bool = True,
    seed: int = 42,
    device: Optional[str] = None,
):
    results = {}

    for layer_idx in layers:
        res = run_single_layer_position_probe(
            runner=runner,
            texts=texts,
            layer_idx=layer_idx,
            max_position=max_position,
            build_batch_size=build_batch_size,
            probe_batch_size=probe_batch_size,
            num_epochs=num_epochs,
            lr=lr,
            remove_special_tokens=remove_special_tokens,
            seed=seed,
            device=device,
        )
        results[layer_idx] = res["test_metrics"]
        print(f"Layer {layer_idx}: {res['test_metrics']}")

    return results


def run_probe_for_selected_layers_with_splits(
    runner: BertHiddenStateRunner,
    train_texts: List[str],
    val_texts: List[str],
    test_texts: List[str],
    layers: Tuple[int, ...] = (1, 6, 12),
    max_position: int = 300,
    build_batch_size: int = 16,
    probe_batch_size: int = 128,
    num_epochs: int = 10,
    lr: float = 1e-3,
    remove_special_tokens: bool = True,
    device: Optional[str] = None,
):
    """
    Run probes for multiple layers using explicit train/validation/test text
    splits from an external dataset.
    """
    results = {}

    for layer_idx in layers:
        res = run_single_layer_position_probe_with_splits(
            runner=runner,
            train_texts=train_texts,
            val_texts=val_texts,
            test_texts=test_texts,
            layer_idx=layer_idx,
            max_position=max_position,
            build_batch_size=build_batch_size,
            probe_batch_size=probe_batch_size,
            num_epochs=num_epochs,
            lr=lr,
            remove_special_tokens=remove_special_tokens,
            device=device,
        )
        results[layer_idx] = res["test_metrics"]
        print(f"Layer {layer_idx}: {res['test_metrics']}")

    return results
