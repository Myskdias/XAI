from typing import List, Optional

import torch
from transformers import AutoModel, AutoTokenizer

from .types import BertBatchOutput


class BertHiddenStateRunner:
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: Optional[str] = None,
        max_length: int = 128,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def run_batch(self, texts: List[str]) -> BertBatchOutput:
        """
        Tokenize and run a batch through BERT.
        Returns raw hidden states and token strings.
        """
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = [h.detach().cpu() for h in outputs.hidden_states]
        input_ids_cpu = input_ids.detach().cpu()
        attention_mask_cpu = attention_mask.detach().cpu()

        tokens = [
            self.tokenizer.convert_ids_to_tokens(seq.tolist())
            for seq in input_ids_cpu
        ]

        return BertBatchOutput(
            texts=texts,
            input_ids=input_ids_cpu,
            attention_mask=attention_mask_cpu,
            tokens=tokens,
            hidden_states=hidden_states,
        )
