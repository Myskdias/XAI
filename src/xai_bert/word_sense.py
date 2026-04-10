from typing import Dict, List, Optional
import re

import torch
import torch.nn.functional as F

from .modeling import BertHiddenStateRunner
from .clipping import clip_embedding_dimensions

def _find_subsequence(sequence: List[int], subsequence: List[int]) -> int:
    if not subsequence:
        return -1
    n = len(subsequence)
    for i in range(len(sequence) - n + 1):
        if sequence[i : i + n] == subsequence:
            return i
    return -1


def contextual_word_vector(
    runner: BertHiddenStateRunner,
    sentence: str,
    word: str,
    layer_idx: int = 12,
    clip_dims: Optional[List[int]] = None,
) -> Optional[torch.Tensor]:
    enc = runner.tokenizer(
        sentence,
        truncation=True,
        max_length=runner.max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    target = word.strip()
    if not target:
        return None

    ids = enc["input_ids"]
    offsets = enc["offset_mapping"][0].tolist()
    start = -1
    end = -1

    # Prefer char-span alignment from sentence text.
    pattern = re.compile(r"\b" + re.escape(target) + r"\b", flags=re.IGNORECASE)
    matches = list(pattern.finditer(sentence))
    if not matches:
        matches = list(re.finditer(re.escape(target), sentence, flags=re.IGNORECASE))

    for m in matches:
        cs, ce = m.start(), m.end()
        token_ids = [
            i
            for i, (s, e) in enumerate(offsets)
            if e > s and not (e <= cs or s >= ce)
        ]
        if token_ids:
            start, end = token_ids[0], token_ids[-1] + 1
            break

    # Fallback: exact sub-token sequence search.
    if start < 0:
        for candidate in (target, target.lower()):
            word_ids = runner.tokenizer(candidate, add_special_tokens=False)["input_ids"]
            if not word_ids:
                continue
            s = _find_subsequence(ids[0].tolist(), word_ids)
            if s >= 0:
                start = s
                end = s + len(word_ids)
                break

    if start < 0:
        return None

    with torch.no_grad():
        out = runner.model(
            input_ids=ids.to(runner.device),
            attention_mask=enc["attention_mask"].to(runner.device),
        )
    hidden = out.hidden_states[layer_idx][0].detach().cpu()
    hidden = hidden[start:end]
    if clip_dims is not None:
        hidden = clip_embedding_dimensions(hidden, dims_to_clip=clip_dims)
    return hidden.mean(dim=0)


def evaluate_wic_with_threshold(
    runner: BertHiddenStateRunner,
    examples: List[Dict[str, object]],
    layer_idx: int = 12,
    threshold: float = 0.7,
    max_examples: Optional[int] = None,
    dims_to_clip: Optional[List[int]] = None,
) -> Dict[str, float]:
    used = 0
    correct = 0
    skipped = 0

    for ex in examples[:max_examples] if max_examples else examples:
        label = ex.get("label", None)
        if label is None:
            continue

        v1 = contextual_word_vector(runner, ex["sentence1"], ex["word"], layer_idx, clip_dims=dims_to_clip[layer_idx] if dims_to_clip else None)
        v2 = contextual_word_vector(runner, ex["sentence2"], ex["word"], layer_idx, clip_dims=dims_to_clip[layer_idx] if dims_to_clip else None)
        if v1 is None or v2 is None:
            skipped += 1
            continue

        sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
        pred = 1 if sim >= threshold else 0
        correct += int(pred == int(label))
        used += 1

    acc = (correct / used) if used > 0 else 0.0
    return {
        "threshold": threshold,
        "layer": layer_idx,
        "used": used,
        "skipped": skipped,
        "accuracy": acc,
    }
