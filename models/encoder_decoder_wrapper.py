import torch
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

    def infer(self, x, meta):
        attempt = 0
        inference_succ = False
        while not inference_succ:
            if attempt > 2:
                break
            try:
                seg_logits = self.model.inference(x, [meta], rescale=True)
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
        net['preds'] = net['logits'].argmax(dim=1)
        return net
