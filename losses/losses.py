import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss


class CosineEmbeddingLossV2(nn.CosineEmbeddingLoss):
    def forward(self, input1: Tensor, input2: Tensor, target=torch.tensor(1)) -> Tensor:
        return super().forward(input1, input2, target)


class LinfLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return torch.linalg.norm(input - target, ord=float('inf'), dim=-1)
