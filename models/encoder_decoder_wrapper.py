from typing import Dict
import sys
import torch.nn as nn
import logging
import time
sys.path.insert(0, "./mmsegmentation")
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder

class EncoderDecoderWrapper(nn.Module):
    def __init__(self, model: EncoderDecoder):
        super().__init__()
        self.model = model
        self.logger = logging.getLogger(str(__class__))

    @staticmethod
    def verify_data(data):
        imgs = data['img']
        img_metas = [im._data for im in data['img_metas']][0]
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got f{type(var)}')
        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) != ' + f'num of image meta ({len(img_metas)})')
        # all images in the same aug batch all of the same ori_shape and pad shape
        for img_meta in img_metas:
            ori_shapes = [_['ori_shape'] for _ in img_meta]
            assert all(shape == ori_shapes[0] for shape in ori_shapes)
            img_shapes = [_['img_shape'] for _ in img_meta]
            assert all(shape == img_shapes[0] for shape in img_shapes)
            pad_shapes = [_['pad_shape'] for _ in img_meta]
            assert all(shape == pad_shapes[0] for shape in pad_shapes)

        assert num_augs == 1, 'currently not supporting TTAs'

    @staticmethod
    def unscale(x, minn, maxx):
        x *= (maxx - minn)
        x += minn
        return x

    @staticmethod
    def get_image(data: Dict, rescale):
        if rescale:
            x = data['scaled_img']
        else:
            x = data['img'][0]
        return x.to('cuda')

    @staticmethod
    def get_meta(data: Dict):
        return [im._data for im in data['img_metas']][0][0]

    def infer(self, img, img_meta):
        inference_succ = False
        while not inference_succ:
            try:
                seg_logits = self.model.inference(img, img_meta, rescale=True)
            except RuntimeError as e:
                self.logger.info('Got error {}. waiting 5 seconds to retry...'.format(e))
                time.sleep(5)
            else:
                inference_succ = True
        return seg_logits

    def forward(self, data: Dict, rescale=False):
        self.verify_data(data)
        net = {}
        img = self.get_image(data, rescale)
        if rescale:
            img = self.unscale(img, data['minn'], data['maxx'])
        img_meta = self.get_meta(data)
        net['logits'] = self.infer(img, img_meta)
        net['preds'] = net['logits'].argmax(dim=1)
        return net

    @staticmethod
    def set_scaled_img(data: Dict):
        x = data['img'][0]
        minn = x.min()
        maxx = x.max()
        scaled_x = (x - minn) / (maxx - minn)
        data['scaled_img'] = scaled_x
        data['minn'] = minn
        data['maxx'] = maxx
