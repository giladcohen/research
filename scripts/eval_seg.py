"""Evaluating DeepLab model with IoU metric"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
from typing import Dict
import torch
import torch.backends.cudnn as cudnn
import mmcv
from mmcv.runner import load_checkpoint, wrap_fp16_model
import argparse

import sys
sys.path.insert(0, "./mmsegmentation")

from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from research.utils import set_logger, tensor2imgs
from research.models.encoder_decoder_wrapper import EncoderDecoderWrapper
from research.datasets.pascal_utils import set_scaled_img, unscale, parse_data, verify_data, show_rgb_img


parser = argparse.ArgumentParser(description='PyTorch PASCAL VOC evaluator')
parser.add_argument('--config',
                    default='/home/gilad/workspace/mmsegmentation/configs/deeplabv3/deeplabv3_r50-d8_512x512_40k_voc12aug.py',
                    type=str, help='python config file')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/glove_emb/pascal/baseline1',
                    type=str, help='checkpoint dir name')
parser.add_argument('--eval_dir', default='debug', type=str, help='attack directory')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')
EVAL_DIR = os.path.join(args.checkpoint_dir, args.eval_dir)
PRED_DIR = os.path.join(EVAL_DIR, 'preds')
OVERLAP_DIR = os.path.join(EVAL_DIR, 'overlaps')

os.makedirs(EVAL_DIR, exist_ok=True)
log_file = os.path.join(EVAL_DIR, 'log.log')
set_logger(log_file)
logger = logging.getLogger()

# dumping args to txt file
with open(os.path.join(EVAL_DIR, 'eval_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

cfg = mmcv.Config.fromfile(args.config)
if cfg.get('cudnn_benchmark', False):
    cudnn.benchmark = True
cfg.model.pretrained = None
cfg.data.test.test_mode = True

distributed = False
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
json_file = os.path.join(EVAL_DIR, 'eval_{}.json'.format(timestamp))

# build the dataloader
dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=1,  #cfg.data.workers_per_gpu,
    dist=distributed,
    shuffle=False)

# build the model and load checkpoint
cfg.model.train_cfg = None
model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
fp16_cfg = cfg.get('fp16', None)
if fp16_cfg is not None:
    wrap_fp16_model(model)
checkpoint = load_checkpoint(model, CHECKPOINT_PATH, map_location='cpu')
if 'CLASSES' in checkpoint.get('meta', {}):
    model.CLASSES = checkpoint['meta']['CLASSES']
else:
    print('"CLASSES" not found in meta, use dataset.CLASSES instead')
    model.CLASSES = dataset.CLASSES
if 'PALETTE' in checkpoint.get('meta', {}):
    model.PALETTE = checkpoint['meta']['PALETTE']
else:
    print('"PALETTE" not found in meta, use dataset.PALETTE instead')
    model.PALETTE = dataset.PALETTE

# clean gpu memory when starting a new evaluation.
torch.cuda.empty_cache()

# model = MMDataParallel(model, device_ids=[0])
wrapper = EncoderDecoderWrapper(model)
wrapper.cuda()
wrapper.eval()
results = []
prog_bar = mmcv.ProgressBar(len(dataset))

# debug
# batch_idx = 4
# data, targets = list(data_loader)[4]
for batch_idx, data in enumerate(data_loader):
    targets = data['gt_semantic_seg']
    verify_data(data)
    parse_data(data)
    meta = data['meta']

    x = data['x'].cuda()
    out = wrapper(x, meta)
    result = out['preds'].cpu().numpy()
    results.extend(result)

    # dump plots
    x_imgs = tensor2imgs(x, **meta['img_norm_cfg'])
    x_img = x_imgs[0]
    h, w, _ = meta['img_shape']
    img_show = x_img[:h, :w, :]
    ori_h, ori_w = meta['ori_shape'][:-1]
    img_show = mmcv.imresize(img_show, (ori_w, ori_h))
    # show_rgb_img(img_show)

    model.show_result_all(
        img_show,
        result,
        None,
        os.path.join(PRED_DIR, meta['ori_filename']),
        os.path.join(OVERLAP_DIR, meta['ori_filename']),
        palette=dataset.PALETTE,
        opacity=0.5)
    prog_bar.update()

# get metrics:
eval_kwargs = {}
eval_kwargs.update(metric='mIoU')
metric = dataset.evaluate(results, **eval_kwargs)
mmcv.dump(metric, json_file, indent=4)

logger.info('done')
