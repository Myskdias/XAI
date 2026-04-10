import importlib
from typing import Dict, List, Optional, Tuple
import random

def load_sst2_text_splits(
    max_train: Optional[int] = None,
    max_val: Optional[int] = None,
    max_test: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Load SST-2 text splits from GLUE.

    Returns:
        train_texts, val_texts, test_texts

    Notes:
        - GLUE/SST-2 test has no public labels, so this helper uses a held-out
            subset from the train split as test_texts.
        - If use_test_ds=False, test_texts is returned as an empty list.
        - If max_test is 0 or negative, test_texts is returned as an empty list.
    """
    try:
        datasets_module = importlib.import_module("datasets")
        load_dataset = getattr(datasets_module, "load_dataset")
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "The 'datasets' package is required for SST-2 loading. "
            "Install it with: pip install datasets"
        ) from exc

    ds = load_dataset("glue", "sst2")
    train_ds = ds["train"].shuffle(seed=seed)
    val_ds = ds["validation"].shuffle(seed=seed)

    if max_train is None:
        max_train = len(train_ds)
    if max_val is None:
        max_val = len(val_ds)

    max_train = min(max_train, len(train_ds))
    max_val = min(max_val, len(val_ds))

    train_selected = train_ds.select(range(max_train))
    val_selected = val_ds.select(range(max_val))

    if max_test is None or max_test > 0:
        # Build a held-out test split from the remaining train portion.
        train_remaining = train_ds.select(range(max_train, len(train_ds)))

        if max_test is None:
            max_test = min(max_val, len(train_remaining))
        else:
            max_test = min(max_test, len(train_remaining))

        if max_test <= 0:
            test_selected = None
        else:
            test_selected = train_remaining.select(range(max_test))
    else:
        test_selected = None

    train_texts = [row["sentence"] for row in train_selected]
    val_texts = [row["sentence"] for row in val_selected]
    if test_selected is None:
        test_texts = []
    else:
        test_texts = [row["sentence"] for row in test_selected]

    return train_texts, val_texts, test_texts


def context_words_for_text(data_texts: List[str], n_word: int = 1000, n_sentence_min: int = 10, random_select: bool = False, seed: Optional[int] = 42) -> List[str]:
    """
    Get n_word words in data_texts such that they appear in at least n_sentence_min sentences.
    """
    word_counter = {}

    for text in data_texts:
        words = set(text.split())
        for word in words:
            word_counter[word] = word_counter.get(word, 0) + 1

    # Filter words that appear in at least n_sentence_min sentences
    filtered_words = {word: count for word, count in word_counter.items() if count >= n_sentence_min}

    if random_select:
        random.seed(seed)
        return random.sample(list(filtered_words.keys()), min(n_word, len(filtered_words)))
    else: # select by frequency
        return sorted(filtered_words.keys(), key=lambda x: filtered_words[x], reverse=True)[:n_word]


def load_wic_splits(
    max_train: Optional[int] = None,
    max_val: Optional[int] = None,
    max_test: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    """Load SuperGLUE/WiC and return train/val/test example dicts."""
    try:
        datasets_module = importlib.import_module("datasets")
        load_dataset = getattr(datasets_module, "load_dataset")
    except (ImportError, AttributeError) as exc:
        raise ImportError("The 'datasets' package is required. Install with: pip install datasets") from exc

    ds = load_dataset("super_glue", "wic")

    def _pick(split_name: str, n: Optional[int]):
        split = ds[split_name].shuffle(seed=seed)
        return split if n is None else split.select(range(min(n, len(split))))

    def _rows_to_examples(rows):
        out = []
        for row in rows:
            label = row.get("label", None)
            if label == -1:
                label = None
            out.append(
                {
                    "word": row["word"],
                    "sentence1": row["sentence1"],
                    "sentence2": row["sentence2"],
                    "label": label,
                }
            )
        return out

    train_rows = _pick("train", max_train)
    val_rows = _pick("validation", max_val)
    test_rows = _pick("test", max_test)
    return _rows_to_examples(train_rows), _rows_to_examples(val_rows), _rows_to_examples(test_rows)