import torch
import torch.backends.cudnn as cudnn
import numpy as np
import argparse
import os
import logging
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "./influence_functions")

from research.datasets.train_val_test_data_loaders import get_loader_with_specific_inds, get_test_loader
from research.utils import boolean_string, pytorch_evaluate, set_logger, get_image_shape
from research.models.utils import get_strides, get_conv1_params, get_model
import influence_functions.pytorch_influence_functions as ptif

# parser = argparse.ArgumentParser(description='Influence functions tutorial using pytorch')
# parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/mi/cifar10/resnet18/s_1k_wo_aug_act_swish', type=str, help='checkpoint dir')
# parser.add_argument('--checkpoint_file', default='ckpt.pth', type=str, help='checkpoint path file name')
# parser.add_argument('--output_dir', default='debug', type=str, help='checkpoint path file name')
# parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
# parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')
#
# args = parser.parse_args()

config = ptif.get_default_config()
checkpoint_dir = '/data/gilad/logs/mi/cifar10/resnet18/s_1k_wo_aug_act_swish'
CHECKPOINT_PATH = os.path.join(checkpoint_dir, 'ckpt.pth')
OUTPUT_DIR = os.path.join(checkpoint_dir, 'influence_functions_debug')
os.makedirs(os.path.join(OUTPUT_DIR), exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, 'log.log')
set_logger(log_file)
logger = logging.getLogger()

logger.info('==> Building model..')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net_cls = get_model('resnet18')
conv1 = get_conv1_params('cifar10')
strides = get_strides('cifar10')
net = net_cls(num_classes=10, activation='swish', conv1=conv1, strides=strides, field='probs')
net = net.to('cuda')
global_state = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))
if 'best_net' in global_state:
    global_state = global_state['best_net']
net.load_state_dict(global_state)
net.eval()
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Data
logger.info('==> Preparing data..')
all_train_inds = np.arange(50000)
train_inds = np.load(os.path.join(checkpoint_dir, 'train_inds.npy'))
val_inds   = np.load(os.path.join(checkpoint_dir, 'val_inds.npy'))
unused_train_inds = np.asarray([i for i in all_train_inds if i not in np.concatenate((train_inds, val_inds))])

train_loader = get_loader_with_specific_inds(
    dataset='cifar10',
    dataset_args=dict(),
    batch_size=100,
    is_training=False,
    indices=train_inds,
    num_workers=0,
    pin_memory=device=='cuda'
)
unused_train_loader = get_loader_with_specific_inds(
    dataset='cifar10',
    dataset_args=dict(),
    batch_size=100,
    is_training=False,
    indices=unused_train_inds,
    num_workers=0,
    pin_memory=device=='cuda'
)
val_loader = get_loader_with_specific_inds(
    dataset='cifar10',
    dataset_args=dict(),
    batch_size=100,
    is_training=False,
    indices=val_inds,
    num_workers=0,
    pin_memory=device=='cuda'
)
test_loader = get_test_loader(
    dataset='cifar10',
    dataset_args=dict(),
    batch_size=100,
    num_workers=0,
    pin_memory=device=='cuda'
)

config['outdir'] = OUTPUT_DIR
influences, harmful, helpful, test_id_num = ptif.calc_influence_single(
    model=net,
    train_loader=train_loader,
    test_loader=test_loader,
    test_id_num=0,
    gpu=0,
    recursion_depth=5000,
    r=1,
    time_logging=True)
