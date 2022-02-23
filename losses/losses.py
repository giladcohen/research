import contextlib
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

class KLDivAbsLoss(nn.KLDivLoss):
    EPS = 1e-8
    def forward(self, input: Tensor, target: Tensor, y: Tensor) -> Tensor:
        in1 = F.normalize(torch.abs(input - y), p=1, dim=1)
        in2 = F.normalize(torch.abs(target - y), p=1, dim=1)
        return super().forward(torch.log(in1 + self.EPS), in2)

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
    elif criterion == 'kl_abs':
        loss_criterion = KLDivAbsLoss(reduction='batchmean')
    else:
        raise AssertionError('Unknown criterion {}'.format(criterion))
    return loss_criterion


class TradesLoss(_Loss):
    def __init__(self, model: nn.Module, eps: float, eps_step: float,
                 steps: int, beta: float, field: str, criterion: str, adv_criterion: str,
                 size_average=None, reduce=None, reduction: str = 'none') -> None:
        super().__init__(size_average, reduce, reduction)
        self.model = model
        self.eps = eps
        self.eps_step = eps_step
        self.steps = steps
        self.beta = beta
        self.field = field
        self.loss_criterion = loss_critetion_factory(criterion)
        self.adv_loss_criterion = loss_critetion_factory(adv_criterion)

    def forward(self, x_natural: Tensor, y: Tensor, kwargs) -> Tuple[Dict, Dict[str, torch.Tensor]]:
        is_training = kwargs['is_training']
        losses = {}
        self.model.eval()
        batch_size = len(x_natural)
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
        with torch.enable_grad():
            outputs = self.model(x_natural)
            out_natural = outputs[self.field]
            for _ in range(self.steps):
                x_adv.requires_grad_()
                out_adv = self.model(x_adv)[self.field]
                if isinstance(self.adv_loss_criterion, KLDivAbsLoss):
                    loss = batch_size * self.adv_loss_criterion(out_adv, out_natural, y)
                else:
                    loss = batch_size * self.adv_loss_criterion(out_adv, out_natural)  # multiplying in batch_size to match reference
                grad = torch.autograd.grad(loss, [x_adv])[0]
                x_adv = x_adv.detach() + self.eps_step * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - self.eps), x_natural + self.eps)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)

        self.model.train(is_training)
        x_adv.requires_grad_(False)
        # zero gradient
        self.model.zero_grad()
        # calculate robust loss
        out_adv = self.model(x_adv)[self.field]
        loss_natural = self.loss_criterion(out_natural, y)
        if isinstance(self.adv_loss_criterion, KLDivAbsLoss):
            loss_robust = self.adv_loss_criterion(out_adv, out_natural, y)
        else:
            loss_robust = self.adv_loss_criterion(out_adv, out_natural)
        loss = loss_natural + self.beta * loss_robust
        losses['natural'] = loss_natural
        losses['robust'] = loss_robust
        losses['loss'] = loss
        return outputs, losses

@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):
    def __init__(self, model: nn.Module, field: str, criterion: str, adv_criterion: str,
                 beta: float, xi: float = 10.0, eps: float = 1.0, steps: int = 1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.model = model
        self.criterion = loss_critetion_factory(criterion)
        self.adv_loss_criterion = loss_critetion_factory(adv_criterion)
        self.field = field
        self.beta = beta
        self.xi = xi
        self.eps = eps
        self.steps = steps

    def forward(self, x_natural: torch.Tensor, y: torch.Tensor, kwargs):
        is_training = kwargs['is_training']
        losses = {}
        self.model.eval()
        # prepare random unit tensor
        d = torch.rand(x_natural.shape).sub(0.5).to(x_natural.device)
        d = _l2_normalize(d)
        with torch.enable_grad():
            outputs = self.model(x_natural)
            out_natural = outputs[self.field]
            with _disable_tracking_bn_stats(self.model):
                # calc adversarial direction
                for _ in range(self.steps):
                    d.requires_grad_()
                    out_adv = self.model(x_natural + self.xi * d)[self.field]
                    adv_distance = self.adv_loss_criterion(out_adv, out_natural.detach())
                    adv_distance.backward()
                    d = _l2_normalize(d.grad)
                    self.model.zero_grad()

        self.model.train(is_training)
        x_adv = x_natural + d * self.eps  # x_natural + r_adv
        self.model.zero_grad()
        out_adv = self.model(x_adv)[self.field]
        loss_natural = self.criterion(out_natural, y)
        loss_robust = self.adv_loss_criterion(out_adv, out_natural)  # calc LDS
        loss = loss_natural + self.beta * loss_robust
        losses['natural'] = loss_natural
        losses['robust'] = loss_robust
        losses['loss'] = loss
        return outputs, losses

class GuidedAdversarialTrainingLoss(_Loss):
    def __init__(self, model: nn.Module, field: str, criterion: str, adv_criterion: str, bern_eps: float, eps: float,
                 eps_step: float, steps: int, l2_reg: float):
        super().__init__(reduction='none')
        self.model = model
        self.bern_eps = bern_eps
        self.eps = eps
        self.eps_step = eps_step
        self.steps = steps
        self.l2_reg = l2_reg
        self.field = field
        self.loss_criterion = loss_critetion_factory(criterion)
        self.adv_loss_criterion = loss_critetion_factory(adv_criterion)

    def forward(self, x_natural: Tensor, y: Tensor, kwargs) -> Tuple[Dict, Dict[str, torch.Tensor]]:
        alt = kwargs['alt']
        is_training = kwargs['is_training']
        losses = {}
        self.model.eval()

        # initializing x_adv
        x_adv = x_natural + self.bern_eps * torch.sign(torch.tensor(0.5) - torch.rand_like(x_natural))
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        with torch.enable_grad():
            outputs = self.model(x_natural)
            out_natural = outputs[self.field]
            # x_adv generation
            for _ in range(self.steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    adv_outputs = self.model(x_adv)
                    out_adv = adv_outputs[self.field]
                    loss_attack = self.adv_loss_criterion(out_adv, y)
                    if alt == 1:
                        loss_l2_reg = self.l2_reg * (torch.linalg.norm(outputs['probs'] - adv_outputs['probs'], dim=1) ** 2.0).mean(0)
                    else:
                        loss_l2_reg = 0.0
                    loss = loss_attack + loss_l2_reg
                    grad = torch.autograd.grad(loss, [x_adv])[0]
                    x_adv = x_adv.detach() + self.eps_step * torch.sign(grad.detach())
                    x_adv = torch.min(torch.max(x_adv, x_natural - self.eps), x_natural + self.eps)
                    x_adv = torch.clamp(x_adv, 0.0, 1.0)

        # adv training
        self.model.train(is_training)
        # zero gradient
        self.model.zero_grad()
        # calculate robust loss
        adv_probs = self.model(x_adv)['probs']
        loss_natural = self.loss_criterion(out_natural, y)
        loss_robust = self.l2_reg * (torch.linalg.norm(outputs['probs'] - adv_probs, dim=1) ** 2.0).mean(0)
        loss = loss_natural + loss_robust
        losses['natural'] = loss_natural
        losses['robust'] = loss_robust
        losses['loss'] = loss
        return outputs, losses

