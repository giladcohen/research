"""Evaluating DeepLab model with IoU metric"""
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
import torch
import torch.backends.cudnn as cudnn
import mmcv
from mmcv import tensor2imgs
from mmcv.runner import load_checkpoint, wrap_fp16_model

import sys
sys.path.insert(0, "./mmsegmentation")

from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.models.losses import CrossEntropyLoss
from research.utils import set_logger
from research.models.encoder_decoder_wrapper import EncoderDecoderWrapper

EPS = 0.031

# get config:
CONFIG_FILE = '/home/gilad/workspace/mmsegmentation/configs/deeplabv3/deeplabv3_r50-d8_512x512_40k_voc12aug.py'
CHECKPOINT_PATH = '/data/gilad/logs/glove_emb/pascal/baseline1/ckpt.pth'
METRICS_DIR = '/data/gilad/logs/glove_emb/pascal/baseline1/fgsm_eps_0.031'
ATTACK_DIR = '/data/gilad/logs/glove_emb/pascal/baseline1/fgsm_eps_0.031/results'

os.makedirs(METRICS_DIR, exist_ok=True)
log_file = os.path.join(METRICS_DIR, 'log.log')
set_logger(log_file)
logger = logging.getLogger()

cfg = mmcv.Config.fromfile(CONFIG_FILE)
if cfg.get('cudnn_benchmark', False):
    cudnn.benchmark = True
cfg.model.pretrained = None
cfg.data.test.test_mode = True

distributed = False
os.makedirs(METRICS_DIR, exist_ok=True)
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
json_file = os.path.join(METRICS_DIR, 'eval_{}.json'.format(timestamp))

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
eval_kwargs = {}

# model = MMDataParallel(model, device_ids=[0])
wrapper = EncoderDecoderWrapper(model)
wrapper.to('cuda')
wrapper.eval()
results = []
adv_results = []
prog_bar = mmcv.ProgressBar(len(dataset))

ce_loss = CrossEntropyLoss()

# debug
# batch_idx = 1
# data, targets = list(data_loader)[1]

def scale(x):
    minn = x.min()
    maxx = x.max()
    scaled_x = (x - minn) / (maxx - minn)
    return scaled_x, minn, maxx


for batch_idx, (data, targets) in enumerate(data_loader):
    # scaling the image in [0, 1]:
    data['scaled_img'], data['minn'], data['maxx'] = scale(data['img'][0])
    data['scaled_img'].requires_grad = True
    targets = targets.to('cuda').long()
    out = wrapper(data)
    kwargs = {'ignore_index': 255}
    loss = ce_loss(out['logits'], targets, **kwargs)

    # attack:
    # x_adv = data['img'][0][0].clone()

    wrapper.zero_grad()
    loss.backward()
    grads = data['scaled_img'].grad
    grads = torch.sign(grads)
    perturbation_step = EPS * grads
    scaled_x_adv = data['scaled_img'] + perturbation_step
    scaled_x_adv = torch.clip(scaled_x_adv, 0.0, 1.0)
    data['scaled_img'] = scaled_x_adv.detach()
    out = wrapper(data)
    result = out['preds'].cpu().numpy()

    # dump plots
    img = wrapper.unscale(scaled_x_adv, data['minn'], data['maxx']).detach()  # data['img'][0]
    img_meta = [im._data for im in data['img_metas']][0][0]
    imgs = tensor2imgs(img, **img_meta[0]['img_norm_cfg'])
    assert len(imgs) == len(img_meta)

    img = imgs[0]
    img_meta = img_meta[0]
    h, w, _ = img_meta['img_shape']
    img_show = img[:h, :w, :]

    ori_h, ori_w = img_meta['ori_shape'][:-1]
    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

    out_file = os.path.join(ATTACK_DIR, img_meta['ori_filename'])
    model.show_result(
        img_show,
        result,
        palette=dataset.PALETTE,
        show=True,
        out_file=out_file,
        opacity=0.5)

    prog_bar.update()


# get metrics:
eval_kwargs.update(metric='mIoU')
metric = dataset.evaluate(results, **eval_kwargs)
mmcv.dump(metric, json_file, indent=4)

logger.info('done')
