import torch
import numpy as np
from typing import Dict
import sys
import torch.nn as nn
import logging
import time
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, "./mmsegmentation")
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmcv.utils.config import ConfigDict

class ArgmaxInterpreter(nn.Module):
    def __init__(self):
        super().__init__()
        print('Constructing Argmax interpreter')

    def forward(self, x):
        return x.argmax(axis=1)

class KNNInterpreter(nn.Module):
    def __init__(self, p, idx_to_vec_path):
        super().__init__()
        print('Constructing KNN interpreter with p={}'.format(p))
        class_emb_vecs = np.load(idx_to_vec_path)
        self.knn = NearestNeighbors(n_neighbors=1, algorithm='brute', p=p)
        self.knn.fit(class_emb_vecs)

    def forward(self, x):
        N, C, H, W = x.shape
        x = x.reshape(-1, C)
        seg_preds = self.knn.kneighbors(x, return_distance=False).squeeze()
        seg_preds = seg_preds.reshape((N, H, W))
        return seg_preds

class CosineInterpreter(nn.Module):
    def __init__(self, idx_to_vec_path):
        super().__init__()
        print('Constructing Cosine interpreter')
        self.class_emb_vecs = np.load(idx_to_vec_path)
        self.cos = nn.CosineSimilarity()
        self.num_classes = self.class_emb_vecs.shape[0]

    def forward(self, x):
        N, C, H, W = x.shape
        x = x.reshape(-1, C)
        x = torch.from_numpy(x)
        distance_mat = torch.zeros((x.size(0), self.num_classes))
        for cls_idx in range(self.num_classes):
            embs = np.tile(self.class_emb_vecs[cls_idx], (N * H * W, 1))
            embs = torch.from_numpy(embs)
            distance_mat[:, cls_idx] = self.cos(x, embs)
        distance_mat = distance_mat.cpu().numpy()
        seg_preds = distance_mat.argmax(1)
        seg_preds = seg_preds.reshape((N, H, W))
        return seg_preds

class EncoderDecoderWrapper(nn.Module):
    def __init__(self, cfg: ConfigDict, model: EncoderDecoder):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.logger = logging.getLogger(str(__class__))
        self.interpreter = self.get_interpreter(cfg.decode_head.loss_decode)
        self.apply_softmax = cfg.decode_head.loss_decode['loss_type'] == 'CrossEntropyLoss'

    @staticmethod
    def get_norm(x: str):
        if x == 'L1':
            return 1
        elif x == 'L2':
            return 2
        elif x == 'Linf':
            return np.inf
        else:
            raise AssertionError('Expecting L1/L2/Linf')

    def get_interpreter(self, cfg_loss):
        if cfg_loss.type == 'CrossEntropyLoss':
            return ArgmaxInterpreter()
        elif cfg_loss.type == 'DistanceLoss':
            if cfg_loss.loss_type in ['L1', 'L2', 'Linf']:
                return KNNInterpreter(self.get_norm(cfg_loss.loss_type), cfg_loss.idx_to_vec_path)
            elif cfg_loss.loss_type == 'cosine':
                return CosineInterpreter(cfg_loss.idx_to_vec_path)
            else:
                raise AssertionError('Expecting L1/L2/Linf or cosine')
        else:
            raise AssertionError('Expecting CrossEntropyLoss or DistanceLoss only')

    def infer(self, x, meta):
        attempt = 0
        inference_succ = False
        while not inference_succ:
            if attempt > 2:
                break
            try:
                seg_logits = self.model.inference(x, [meta], rescale=True, softmax=self.apply_softmax)
            except RuntimeError as e:
                self.logger.info('Got error {}. waiting 5 seconds to retry...'.format(e))
                time.sleep(5)
                attempt += 1
            else:
                inference_succ = True
        if inference_succ:
            return seg_logits
        else:
            raise AssertionError(e)

    def forward(self, x: torch.tensor, meta: Dict):
        net = {}
        net['logits'] = self.infer(x, meta)
        net['preds'] = self.interpreter(net['logits'].detach().cpu().numpy())
        return net
