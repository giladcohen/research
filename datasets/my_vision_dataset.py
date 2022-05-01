from typing import Any, Tuple
import torch
from torchvision.datasets import VisionDataset
import numpy as np

#debug:
import matplotlib.pyplot as plt
from research.utils import convert_tensor_to_image
from time import time

class MyVisionDataset(VisionDataset):

    def __init__(self, data: np.ndarray, y_gt: np.ndarray, *args, **kwargs) -> None:
        root = None
        super().__init__(root, *args, **kwargs)
        self.data = torch.from_numpy(np.expand_dims(data, 0))
        self.y_gt = torch.from_numpy(np.expand_dims(y_gt, 0))
        assert type(self.data) == type(self.y_gt) == torch.Tensor, \
            'types of data, y_gt must be tensor type'
        self.img_shape = tuple(self.data.size()[1:])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        assert index == 0
        img, y_gt = self.data[index], self.y_gt[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            y_gt = self.target_transform(y_gt)

        return img, y_gt
