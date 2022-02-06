import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss
from typing import Tuple, Dict
from torch.autograd import Variable


class CosineEmbeddingLossV2(nn.CosineEmbeddingLoss):
    def forward(self, input1: Tensor, input2: Tensor, weights=None) -> Tensor:
        if weights is None:
            assert self.reduction == 'mean', 'Expecting reduction=mean but we have reduction={}'.format(self.reduction)
            return super().forward(input1, input2, torch.ones(input1.size(0), device=input1.device))
        else:
            assert self.reduction == 'none', 'Expecting reduction=none but we have reduction={}'.format(self.reduction)
            cosine_loss = super().forward(input1, input2, torch.ones(input1.size(0), device=input1.device))
            return (weights * cosine_loss).mean()

class L1Loss(_Loss):
    def forward(self, input: Tensor, target: Tensor, weights=None) -> Tensor:
        if weights is None:
            diffs = torch.linalg.norm(input - target, ord=1, dim=-1)
        else:
            diffs = weights * torch.linalg.norm(input - target, ord=1, dim=-1)
        return diffs.mean()

class L2Loss(_Loss):
    def forward(self, input: Tensor, target: Tensor, weights=None) -> Tensor:
        if weights is None:
            diffs = torch.linalg.norm(input - target, ord=2, dim=-1)
        else:
            diffs = weights * torch.linalg.norm(input - target, ord=2, dim=-1)
        return diffs.mean()

class LinfLoss(_Loss):
    def forward(self, input: Tensor, target: Tensor, weights=None) -> Tensor:
        if weights is None:
            diffs = torch.linalg.norm(input - target, ord=float('inf'), dim=-1)
        else:
            diffs = weights * torch.linalg.norm(input - target, ord=float('inf'), dim=-1)
        return diffs.mean()

class KLDivLossV2(nn.KLDivLoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        in1 = F.log_softmax(input, dim=1)
        in2 = F.softmax(target, dim=1)
        return super().forward(in1, in2)

class TradesLoss(_Loss):
    @staticmethod
    def loss_critetion_factory(criterion: str):
        if criterion == 'L1':
            loss_criterion = L1Loss()
        elif criterion == 'L2':
            loss_criterion = L2Loss()
        elif criterion == 'Linf':
            loss_criterion = LinfLoss()
        elif criterion == 'cosine':
            loss_criterion = CosineEmbeddingLossV2()
        elif criterion == 'ce':
            loss_criterion = nn.CrossEntropyLoss()
        elif criterion == 'kl':
            loss_criterion = KLDivLossV2(reduction='batchmean')
        else:
            raise AssertionError('Unknown criterion {}'.format(criterion))
        return loss_criterion

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, step_size: float, epsilon,
                 perturb_steps: int, beta: float, field: str, criterion: str, adv_criterion: str,
                 size_average=None, reduce=None, reduction: str = 'none') -> None:
        super().__init__(size_average, reduce, reduction)
        self.model = model
        self.optimizer = optimizer
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.beta = beta
        self.field = field
        self.loss_criterion = self.loss_critetion_factory(criterion)
        self.adv_loss_criterion = self.loss_critetion_factory(adv_criterion)

    def forward(self, x_natural: Tensor, y: Tensor, is_training=False) -> Tuple[Dict, Dict[str, torch.Tensor]]:
        losses = {}
        self.model.eval()
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
        for _ in range(self.perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                out_adv = self.model(x_adv)[self.field]
                out_natural = self.model(x_natural)[self.field]
                loss = self.adv_loss_criterion(out_adv, out_natural)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        self.model.train(is_training)

        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_adv.requires_grad_(False)
        # zero gradient
        self.optimizer.zero_grad()
        # calculate robust loss
        outputs = self.model(x_natural)
        out_natural = outputs[self.field]
        out_adv = self.model(x_adv)[self.field]
        loss_natural = self.loss_criterion(out_natural, y)
        loss_robust = self.adv_loss_criterion(out_adv, out_natural)
        loss = loss_natural + self.beta * loss_robust
        losses['natural'] = loss_natural
        losses['robust'] = loss_robust
        losses['loss'] = loss
        return outputs, losses
