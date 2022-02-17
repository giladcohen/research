from typing import Tuple, Type, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from robustbench.model_zoo.architectures.dm_wide_resnet import CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD, \
    DMWideResNet

class DMWideResNetV2(DMWideResNet):
    def __init__(self, *args, **kwargs):
        ext_linear = kwargs.pop('ext_linear', None)
        self.use_ext_linear = ext_linear is not None
        super().__init__(*args, **kwargs)
        if self.use_ext_linear:
            self.ext_linear = nn.Linear(self.num_channels, ext_linear)
            self.logits = nn.Linear(ext_linear, kwargs['num_classes'])

    def forward(self, x):
        net = {}
        if self.padding > 0:
            x = F.pad(x, (self.padding,) * 4)
        out = (x - self.mean) / self.std
        out = self.init_conv(out)
        out = self.layer(out)
        out = self.relu(self.batchnorm(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.num_channels)
        net['embeddings'] = out
        if self.use_ext_linear:
            out = self.ext_linear(out)
            net['glove_embeddings'] = out
        out = self.logits(out)
        net['logits'] = out
        net['probs'] = F.softmax(out, dim=1)
        net['preds'] = net['probs'].argmax(dim=1)
        return net
