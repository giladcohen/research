"""Evaluating DeepLab model with IoU metric"""
import os
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
from research.utils import set_logger


# get config:
CONFIG_FILE = '/home/gilad/workspace/mmsegmentation/configs/deeplabv3/deeplabv3_r50-d8_512x512_40k_voc12aug.py'
CHECKPOINT_PATH = '/data/gilad/logs/glove_emb/pascal/baseline1/ckpt.pth'
ATTACK_DIR = '/data/gilad/logs/glove_emb/pascal/baseline1/fgsm'

os.makedirs(ATTACK_DIR, exist_ok=True)
log_file = os.path.join(ATTACK_DIR, 'log.log')
set_logger(log_file)
logger = logging.getLogger()

cfg = mmcv.Config.fromfile(CONFIG_FILE)
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
eval_kwargs = {}

# model = MMDataParallel(model, device_ids=[0])
model.to('cuda')
model.eval()
results = []
prog_bar = mmcv.ProgressBar(len(dataset))
loader_indices = data_loader.batch_sampler

# for batch_indices, data in zip(loader_indices, data_loader):
batch_indices = list(loader_indices)[0]
data = list(data_loader)[0]
imgs = data['img']
img_metas = data['img_metas']
img_metas = [im._data for im in img_metas][0]

# assertions
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
img = imgs[0].to('cuda')
img_meta = img_metas[0]

inference_succ = False
while not inference_succ:
    try:
        seg_logits = model.inference(img, img_meta, rescale=True)
    except RuntimeError as e:
        logger.info('Got error {}. waiting 5 seconds to retry...'.format(e))
        time.sleep(5)
    else:
        inference_succ = True

seg_preds = seg_logits.argmax(dim=1)
result = seg_preds.cpu().numpy()
results.extend(result)

# dump plots
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
