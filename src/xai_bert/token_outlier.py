from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch

from .embeddings import flatten_valid_tokens
from .modeling import BertHiddenStateRunner


def _dominant_dim(layer_stats: Dict[str, object], mode: str) -> int:
    if mode == "min":
        return int(layer_stats["top_min_dim"])
    if mode == "max":
        return int(layer_stats["top_max_dim"])
    raise ValueError(f"Unsupported mode: {mode}. Use 'min' or 'max'.")


def _event_dims(embeddings: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "min":
        return torch.argmin(embeddings, dim=1)
    if mode == "max":
        return torch.argmax(embeddings, dim=1)
    raise ValueError(f"Unsupported mode: {mode}. Use 'min' or 'max'.")


def counterfactual_mask_drop_for_token(
    runner: BertHiddenStateRunner,
    texts: List[str],
    layer_idx: int,
    token: str,
    dominant_dim: int,
    mode: str = "min",
    remove_special_tokens: bool = True,
    max_texts: int = 300,
) -> Dict[str, float]:
    token_id = runner.tokenizer.convert_tokens_to_ids(token)
    mask_id = runner.tokenizer.mask_token_id
    if token_id is None or mask_id is None:
        return {"counterfactual_occurrences": 0, "before_rate": 0.0, "after_rate": 0.0, "drop": 0.0}

    before_events = 0
    after_events = 0
    occurrences = 0
    used = 0

    with torch.no_grad():
        for text in texts:
            if max_texts and used >= max_texts:
                break
            enc = runner.tokenizer(text, truncation=True, max_length=runner.max_length, return_tensors="pt")
            ids = enc["input_ids"]
            attn = enc["attention_mask"]
            pos = (ids[0] == token_id) & (attn[0] == 1)
            if remove_special_tokens:
                cls_id, sep_id = runner.tokenizer.cls_token_id, runner.tokenizer.sep_token_id
                if cls_id is not None:
                    pos &= ids[0] != cls_id
                if sep_id is not None:
                    pos &= ids[0] != sep_id
            if not pos.any():
                continue
            used += 1

            device_attn = attn.to(runner.device)
            base = runner.model(input_ids=ids.to(runner.device), attention_mask=device_attn).hidden_states[layer_idx][0].detach().cpu()
            before_events += int((_event_dims(base, mode)[pos] == dominant_dim).sum().item())

            cf_ids = ids.clone()
            cf_ids[0, pos] = mask_id
            cf = runner.model(input_ids=cf_ids.to(runner.device), attention_mask=device_attn).hidden_states[layer_idx][0].detach().cpu()
            after_events += int((_event_dims(cf, mode)[pos] == dominant_dim).sum().item())
            occurrences += int(pos.sum().item())

    if occurrences == 0:
        return {"counterfactual_occurrences": 0, "before_rate": 0.0, "after_rate": 0.0, "drop": 0.0}

    before_rate = before_events / occurrences
    after_rate = after_events / occurrences
    return {"counterfactual_occurrences": occurrences, "before_rate": before_rate, "after_rate": after_rate, "drop": before_rate - after_rate}


def detect_token_outlier_responsibility(
    runner: BertHiddenStateRunner,
    texts: List[str],
    outlier_stats: Dict[int, Dict[str, object]],
    layers: Optional[Iterable[int]] = None,
    mode: str = "min",
    batch_size: int = 16,
    remove_special_tokens: bool = True,
    min_occurrences: int = 20,
    top_k: int = 15,
    max_counterfactual_texts: int = 300,
) -> Dict[int, Dict[str, object]]:
    """
    Identify token types most responsible for dominant outlier events in each layer.

        For each layer, this returns baseline outlier ratio and token ranking
        where "drop" is estimated by rerunning BERT after masking the token.
    """
    if not texts:
        raise ValueError("texts must not be empty.")

    selected_layers = sorted(layers) if layers is not None else sorted(outlier_stats.keys())
    dominant_dims = {
        layer_idx: _dominant_dim(outlier_stats[layer_idx], mode=mode)
        for layer_idx in selected_layers
    }

    per_layer = {}
    for layer_idx in selected_layers:
        per_layer[layer_idx] = {
            "token_total": Counter(),
            "token_event": Counter(),
            "position_total": Counter(),
            "position_event": Counter(),
            "n_tokens": 0,
            "n_events": 0,
        }

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch_out = runner.run_batch(batch_texts)

        for layer_idx in selected_layers:
            flat = flatten_valid_tokens(
                batch_output=batch_out,
                layer_idx=layer_idx,
                remove_special_tokens=remove_special_tokens,
            )

            event_dims = _event_dims(flat.embeddings, mode=mode)
            dominant_dim = dominant_dims[layer_idx]
            is_event = event_dims == dominant_dim

            positions = flat.positions.tolist()
            batch_indices = flat.batch_indices.tolist()
            state = per_layer[layer_idx]

            state["n_tokens"] += len(flat.token_strings)
            state["n_events"] += int(is_event.sum().item())

            for i, token in enumerate(flat.token_strings):
                position = int(positions[i])
                event = bool(is_event[i].item())

                state["token_total"][token] += 1
                state["position_total"][position] += 1

                if event:
                    state["token_event"][token] += 1
                    state["position_event"][position] += 1

    results = {}

    for layer_idx in selected_layers:
        state = per_layer[layer_idx]
        n_tokens = state["n_tokens"]
        n_events = state["n_events"]

        if n_tokens == 0:
            raise ValueError(f"No valid tokens for layer {layer_idx}.")

        baseline_ratio = n_events / n_tokens

        token_rows = []
        token_total = state["token_total"]
        token_event = state["token_event"]
        candidate_tokens = [tok for tok, cnt in token_total.items() if cnt >= min_occurrences]
        candidate_tokens.sort(key=lambda tok: token_event[tok], reverse=True)
        candidate_tokens = candidate_tokens[: max(top_k * 4, top_k)]

        for token in candidate_tokens:
            total_count = token_total[token]
            if total_count < min_occurrences:
                continue

            event_count = token_event[token]
            trigger_rate = event_count / total_count if total_count else 0.0
            share_of_events = event_count / n_events if n_events else 0.0
            cf = counterfactual_mask_drop_for_token(
                runner=runner,
                texts=texts,
                layer_idx=layer_idx,
                token=token,
                dominant_dim=dominant_dims[layer_idx],
                mode=mode,
                remove_special_tokens=remove_special_tokens,
                max_texts=max_counterfactual_texts,
            )

            token_rows.append(
                {
                    "token": token,
                    "occurrences": total_count,
                    "event_occurrences": event_count,
                    "trigger_rate": trigger_rate,
                    "share_of_events": share_of_events,
                    "delta_ratio_if_removed": cf["drop"],
                    "counterfactual_occurrences": cf["counterfactual_occurrences"],
                }
            )

        token_rows.sort(
            key=lambda r: (
                r["delta_ratio_if_removed"],
                r["share_of_events"],
                r["trigger_rate"],
                r["event_occurrences"],
            ),
            reverse=True,
        )

        position_rows = []
        for position, total_count in state["position_total"].items():
            event_count = state["position_event"][position]
            position_rows.append(
                {
                    "position": position,
                    "occurrences": total_count,
                    "event_occurrences": event_count,
                    "event_rate": event_count / total_count if total_count else 0.0,
                    "share_of_events": event_count / n_events if n_events else 0.0,
                }
            )

        position_rows.sort(
            key=lambda r: (r["event_rate"], r["share_of_events"], r["occurrences"]),
            reverse=True,
        )

        results[layer_idx] = {
            "mode": mode,
            "dominant_dim": dominant_dims[layer_idx],
            "n_tokens": n_tokens,
            "n_events": n_events,
            "baseline_outlier_ratio": baseline_ratio,
            "top_tokens": token_rows[:top_k],
            "top_positions": position_rows[:top_k],
        }

    return results


def print_token_outlier_summary(
    results: Dict[int, Dict[str, object]],
    top_k: int = 10,
) -> None:
    """
    Pretty-print token responsibility results per layer.
    """
    for layer_idx, res in results.items():
        print(f"Layer {layer_idx} ({res['mode']}, dominant dim={res['dominant_dim']})")
        print(
            f"  tokens={res['n_tokens']}, outlier_events={res['n_events']}, "
            f"baseline_ratio={res['baseline_outlier_ratio']:.4f}"
        )
        print("  Top responsible tokens:")

        for row in res["top_tokens"][:top_k]:
            print(
                f"    {row['token']:<15} occ={row['occurrences']:<6} "
                f"events={row['event_occurrences']:<6} "
                f"trigger={row['trigger_rate']:.3f} "
                f"share={row['share_of_events']:.3f} "
                f"drop={row['delta_ratio_if_removed']:.4f}"
            )

        # print("  Top positions (event rate):")
        # for row in res["top_positions"][:top_k]:
        #     print(
        #         f"    pos={row['position']:<4} occ={row['occurrences']:<6} "
        #         f"events={row['event_occurrences']:<6} "
        #         f"rate={row['event_rate']:.3f}"
        #     )
        print()
