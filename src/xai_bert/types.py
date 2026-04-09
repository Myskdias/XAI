from dataclasses import dataclass
from typing import List

import torch


@dataclass
class BertBatchOutput:
    texts: List[str]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    tokens: List[List[str]]
    hidden_states: List[torch.Tensor]


@dataclass
class FlattenedLayerData:
    embeddings: torch.Tensor
    positions: torch.Tensor
    batch_indices: torch.Tensor
    token_indices: torch.Tensor
    token_strings: List[str]


@dataclass
class PositionProbeDatasetTensors:
    X: torch.Tensor
    y: torch.Tensor
