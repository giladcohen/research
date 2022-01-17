"""Attacking and evaluating DeepLab model with IoU metric"""
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
from mmseg.models.losses import CrossEntropyLoss, DistanceLoss
from research.utils import set_logger, tensor2imgs
from research.models.encoder_decoder_wrapper import EncoderDecoderWrapper
from research.datasets.pascal_utils import set_scaled_img, unscale, parse_data, verify_data, show_rgb_img


parser = argparse.ArgumentParser(description='PyTorch PASCAL VOC adversarial attack')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/glove_emb/pascal/glove_gpus_4x2_L2_lr_0.01_iters_10k_lr_steps_double_conv',
                    type=str, help='checkpoint dir name')
parser.add_argument('--checkpoint_file', default='latest.pth', type=str, help='checkpoint file name')
parser.add_argument('--idx_to_vec_path', default='/data/gilad/logs/glove_emb/pascal/glove_idx_to_emb.npy',
                    type=str, help='path to glove vec npy file')
parser.add_argument('--class_weight', default=None,  # /data/dataset/VOCdevkit/VOC_seg_weights.npy
                    type=str, help='path to class weight npy file')
parser.add_argument('--attack', default='fgsm', type=str, help='attack: fgsm, jsma, pgd, deepfool, cw')
parser.add_argument('--attack_dir', default='debug', type=str, help='attack directory')

# for FGSM/PGD:
parser.add_argument('--eps'     , default=0.031, type=float, help='maximum Linf deviation from original image')
parser.add_argument('--eps_step', default=0.003, type=float, help='step size of each adv iteration')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

CONFIG_PATH = os.path.join(args.checkpoint_dir, 'config.py')
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, args.checkpoint_file)
ATTACK_DIR = os.path.join(args.checkpoint_dir, args.attack_dir)
ADV_IMGS_DIR = os.path.join(ATTACK_DIR, 'adv_images')
PRED_DIR = os.path.join(ATTACK_DIR, 'preds')
OVERLAP_DIR = os.path.join(ATTACK_DIR, 'overlaps')

os.makedirs(ATTACK_DIR, exist_ok=True)
log_file = os.path.join(ATTACK_DIR, 'log.log')
set_logger(log_file)
logger = logging.getLogger()

# dumping args to txt file
with open(os.path.join(ATTACK_DIR, 'attack_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

cfg = mmcv.Config.fromfile(CONFIG_PATH)
if cfg.get('cudnn_benchmark', False):
    cudnn.benchmark = True
cfg.model.pretrained = None
cfg.data.test.test_mode = True

distributed = False
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
json_file = os.path.join(ATTACK_DIR, 'eval_{}.json'.format(timestamp))

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
wrapper = EncoderDecoderWrapper(cfg.model, model)
wrapper.cuda()
wrapper.eval()
results = []
prog_bar = mmcv.ProgressBar(len(dataset))

# setting loss criterion
def get_loss_criterion(cfg_loss):
    if cfg_loss.type == 'CrossEntropyLoss':
        return CrossEntropyLoss(class_weight=args.class_weight)
    elif cfg_loss.type == 'DistanceLoss':
        return DistanceLoss(cfg_loss.loss_type, args.idx_to_vec_path, args.class_weight)
    else:
        raise AssertionError('loss type {} is not supported'.format(cfg_loss.type))


criterion = get_loss_criterion(cfg.model.decode_head.loss_decode)


def generate_fgsm(wrapper, x, meta, targets):
    out = wrapper(x, meta)
    kwargs = {'ignore_index': 255}
    loss = criterion(out['logits'], targets, **kwargs)
    wrapper.zero_grad()
    loss.backward()
    grads = x.grad
    grads = torch.sign(grads)
    perturbation_step = args.eps * grads
    scaled_x_adv = x + perturbation_step
    scaled_x_adv = torch.clip(scaled_x_adv, 0.0, 1.0)
    scaled_x_adv = scaled_x_adv.detach()
    x_adv = unscale(scaled_x_adv, meta['minn'], meta['maxx'])
    return x_adv

# debug
batch_idx = 4
data = list(data_loader)[4]

for batch_idx, data in enumerate(data_loader):
    targets = data['gt_semantic_seg']
    # scaling the image in [0, 1]:
    verify_data(data)
    parse_data(data)
    meta = data['meta']
    set_scaled_img(data)

    x_adv_init = data['scaled_x'].clone()
    x_adv_init = x_adv_init.cuda()
    x_adv_init.requires_grad = True
    targets = targets.cuda().long()

    if args.attack == 'fgsm':
        x_adv = generate_fgsm(wrapper, x_adv_init, meta, targets)
    else:
        err_str = 'attack {} is not supported'.format(args.attack)
        logger.error(err_str)
        raise AssertionError(err_str)

    out = wrapper(x_adv, meta)
    result = out['preds']
    results.extend(result)

    # dump plots
    x_adv_imgs = tensor2imgs(x_adv, **meta['img_norm_cfg'])
    x_adv_img = x_adv_imgs[0]
    h, w, _ = meta['img_shape']
    img_show = x_adv_img[:h, :w, :]
    ori_h, ori_w = meta['ori_shape'][:-1]
    img_show = mmcv.imresize(img_show, (ori_w, ori_h))
    # show_rgb_img(img_show)

    model.show_result_all(
        img_show,
        result,
        os.path.join(ADV_IMGS_DIR, meta['ori_filename']),
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
