from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling


def build_tiny_bert_mlm(
    model_name: str = "prajjwal1/bert-tiny",
    use_positional_embeddings: bool = True,
    device: Optional[str] = None,
):
    """Load tiny BERT MLM and optionally disable positional embeddings."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=True)

    # reset parameters
    model.init_weights()

    pos_emb = model.bert.embeddings.position_embeddings
    if use_positional_embeddings:
        pos_emb.weight.requires_grad = True
    else:
        pos_emb.weight.data.zero_()
        pos_emb.weight.requires_grad = False

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model


def pretrain_tiny_bert_mlm(
    texts: List[str],
    model_name: str = "prajjwal1/bert-tiny",
    use_positional_embeddings: bool = True,
    max_length: int = 64,
    batch_size: int = 16,
    lr: float = 5e-5,
    num_steps: int = 100,
    mlm_probability: float = 0.15,
    device: Optional[str] = None,
) -> Dict[str, object]:
    """Simple MLM pretraining loop on a list of texts."""
    tokenizer, model = build_tiny_bert_mlm(model_name, use_positional_embeddings, device=device)
    enc = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)
    items = [
        {"input_ids": ids, "attention_mask": mask}
        for ids, mask in zip(enc["input_ids"], enc["attention_mask"])
        if sum(mask) > 2
    ]
    if not items:
        raise ValueError("No valid pretraining examples after tokenization.")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability)
    loader = DataLoader(items, batch_size=batch_size, shuffle=True, collate_fn=collator)
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: List[float] = []

    model.train()
    while len(losses) < num_steps:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.item()))
            if len(losses) >= num_steps:
                break

    return {
        "model": model,
        "tokenizer": tokenizer,
        "use_positional_embeddings": use_positional_embeddings,
        "losses": losses,
        "final_loss": losses[-1],
    }


def evaluate_mlm_loss(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 64,
    batch_size: int = 16,
    mlm_probability: float = 0.15,
) -> float:
    """Estimate mean MLM loss on a text list."""
    enc = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)
    items = [{"input_ids": ids, "attention_mask": mask} for ids, mask in zip(enc["input_ids"], enc["attention_mask"])]
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability)
    loader = DataLoader(items, batch_size=batch_size, shuffle=False, collate_fn=collator)

    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            total_loss += float(model(**batch).loss.item())
            total_batches += 1

    return total_loss / max(total_batches, 1)
