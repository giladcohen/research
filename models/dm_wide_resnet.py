from typing import Tuple, Type, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from robustbench.model_zoo.architectures.dm_wide_resnet import CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD, \
    DMWideResNet

class DMWideResNetV2(DMWideResNet):
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
        out = self.logits(out)
        net['logits'] = out
        net['probs'] = F.softmax(out, dim=1)
        net['preds'] = net['probs'].argmax(dim=1)
        return net
