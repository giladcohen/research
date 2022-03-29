import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import argparse
import os
import logging
import sys
import json
import time

sys.path.insert(0, ".")
sys.path.insert(0, "./influence_functions")

from research.datasets.train_val_test_data_loaders import get_loader_with_specific_inds, get_test_loader, \
    get_normalized_tensor, get_dataset_with_specific_records
from research.utils import boolean_string, pytorch_evaluate, set_logger, get_image_shape, get_num_classes,\
    get_max_train_size
from research.models.utils import get_strides, get_conv1_params, get_model
import influence_functions.pytorch_influence_functions as ptif
from influence_functions.pytorch_influence_functions.influence_functions import calc_s_test, calc_grad_z, \
    calc_all_influences

parser = argparse.ArgumentParser(description='Influence functions tutorial using pytorch')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/mi/cifar10/resnet18/s_1k_wo_aug_act_swish', type=str, help='checkpoint dir')
parser.add_argument('--checkpoint_file', default='ckpt.pth', type=str, help='checkpoint path file name')
parser.add_argument('--output_dir', default='influence_functions', type=str, help='checkpoint path file name')
parser.add_argument('--attacker_knowledge', type=float, default=0.5, help='The portion of samples available to the attacker.')
parser.add_argument('--calc_grad_z', type=boolean_string, default=False, help='Calculate grad_z for train inputs')
parser.add_argument('--calc_s_test', type=boolean_string, default=False, help='Calculate s_test for test inputs')
parser.add_argument('--calc_influences', type=boolean_string, default=False, help='Calculate the influence scores for the s_test_set')
parser.add_argument('--s_test_set', type=str, help='set to calculate s_test for: member_train_set/non_member_train_set/member_test_set/non_member_test_set')
parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()
rand_gen = np.random.RandomState(int(time.time()))

with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')
OUTPUT_DIR = os.path.join(args.checkpoint_dir, args.output_dir)
DATA_DIR = os.path.join(args.checkpoint_dir, 'data')

if args.calc_grad_z:
    os.makedirs(os.path.join(OUTPUT_DIR, 'grad_z'), exist_ok=True)
if args.calc_s_test:
    os.makedirs(os.path.join(OUTPUT_DIR, 's_test', args.s_test_set), exist_ok=True)
if args.calc_influences:
    os.makedirs(os.path.join(OUTPUT_DIR, 'influences'), exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

with open(os.path.join(OUTPUT_DIR, 'attack_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

log_file = os.path.join(OUTPUT_DIR, 'log.log')
set_logger(log_file)
logger = logging.getLogger()

dataset = train_args['dataset']
img_shape = get_image_shape(dataset)
num_classes = get_num_classes(dataset)
max_train_size = get_max_train_size(dataset)
batch_size = 100

logger.info('==> Building model..')
net_cls = get_model(train_args['net'])
if 'resnet' in train_args['net']:
    conv1 = get_conv1_params(dataset)
    strides = get_strides(dataset)
    net = net_cls(num_classes=num_classes, activation=train_args['activation'], conv1=conv1, strides=strides, field='logits')
else:
    raise AssertionError('Does not support non Resnet architectures')
net = net.to(device)
global_state = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))
if 'best_net' in global_state:
    global_state = global_state['best_net']
net.load_state_dict(global_state)
net.eval()
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Data
if not os.path.exists(os.path.join(DATA_DIR, 'X_member_train.npy')):
    logger.info('==> Preparing data..')
    all_train_inds = np.arange(50000)
    train_inds = np.load(os.path.join(args.checkpoint_dir, 'train_inds.npy'))
    val_inds   = np.load(os.path.join(args.checkpoint_dir, 'val_inds.npy'))
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
    X_train_member = get_normalized_tensor(train_loader, img_shape, batch_size)
    y_train_member = np.asarray(train_loader.dataset.targets)
    X_train_non_member = get_normalized_tensor(unused_train_loader, img_shape, batch_size)
    y_train_non_member = np.asarray(unused_train_loader.dataset.targets)
    X_val = get_normalized_tensor(val_loader, img_shape, batch_size)
    y_val = np.asarray(val_loader.dataset.targets)
    X_test = get_normalized_tensor(test_loader, img_shape, batch_size)
    y_test = np.asarray(test_loader.dataset.targets)

    del train_loader, unused_train_loader, val_loader, test_loader

    X_member = X_train_member
    y_member = y_train_member
    X_non_member = np.concatenate((X_train_non_member, X_test))
    y_non_member = np.concatenate((y_train_non_member, y_test))
    # building new training and test set
    assert X_non_member.shape[0] >= X_member.shape[0], 'For testing, we require more non members than members'
    # building train/test set for members
    membership_train_size = int(args.attacker_knowledge * X_member.shape[0])
    membership_test_size = X_member.shape[0] - membership_train_size
    train_member_inds = rand_gen.choice(X_member.shape[0], membership_train_size, replace=False)
    train_member_inds.sort()
    X_member_train = X_member[train_member_inds]
    y_member_train = y_member[train_member_inds]

    test_member_inds = np.asarray([i for i in np.arange(X_member.shape[0]) if i not in train_member_inds])
    test_member_inds = rand_gen.choice(test_member_inds, membership_test_size, replace=False)
    test_member_inds.sort()
    X_member_test = X_member[test_member_inds]
    y_member_test = y_member[test_member_inds]
    assert test_member_inds.shape[0] == membership_test_size

    # building train/test set for non members
    non_membership_train_size = membership_train_size
    non_membsership_test_size = membership_test_size
    train_non_member_inds = rand_gen.choice(X_non_member.shape[0], non_membership_train_size, replace=False)
    train_non_member_inds.sort()
    X_non_member_train = X_non_member[train_non_member_inds]
    y_non_member_train = y_non_member[train_non_member_inds]

    test_non_member_inds = np.asarray([i for i in np.arange(X_non_member.shape[0]) if i not in train_non_member_inds])
    test_non_member_inds = rand_gen.choice(test_non_member_inds, non_membsership_test_size, replace=False)
    test_non_member_inds.sort()
    X_non_member_test = X_non_member[test_non_member_inds]
    y_non_member_test = y_non_member[test_non_member_inds]
    assert X_member_test.shape[0] == X_non_member_test.shape[0], 'assert balanced test set for member/non-member'

    np.save(os.path.join(DATA_DIR, 'X_member_train.npy'), X_member_train)
    np.save(os.path.join(DATA_DIR, 'y_member_train.npy'), y_member_train)
    np.save(os.path.join(DATA_DIR, 'X_non_member_train.npy'), X_non_member_train)
    np.save(os.path.join(DATA_DIR, 'y_non_member_train.npy'), y_non_member_train)
    np.save(os.path.join(DATA_DIR, 'X_member_test.npy'), X_member_test)
    np.save(os.path.join(DATA_DIR, 'y_member_test.npy'), y_member_test)
    np.save(os.path.join(DATA_DIR, 'X_non_member_test.npy'), X_non_member_test)
    np.save(os.path.join(DATA_DIR, 'y_non_member_test.npy'), y_non_member_test)
else:
    logger.info('loading data..')
    X_member_train = np.load(os.path.join(DATA_DIR, 'X_member_train.npy'))
    y_member_train = np.load(os.path.join(DATA_DIR, 'y_member_train.npy'))
    X_non_member_train = np.load(os.path.join(DATA_DIR, 'X_non_member_train.npy'))
    y_non_member_train = np.load(os.path.join(DATA_DIR, 'y_non_member_train.npy'))
    X_member_test = np.load(os.path.join(DATA_DIR, 'X_member_test.npy'))
    y_member_test = np.load(os.path.join(DATA_DIR, 'y_member_test.npy'))
    X_non_member_test = np.load(os.path.join(DATA_DIR, 'X_non_member_test.npy'))
    y_non_member_test = np.load(os.path.join(DATA_DIR, 'y_non_member_test.npy'))

tensor_dataset = TensorDataset(torch.from_numpy(X_member_train),
                               torch.from_numpy(y_member_train))
train_loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False,
                          pin_memory=False, drop_last=False)

if args.s_test_set == 'member_train_set':
    pass
elif args.s_test_set == 'non_member_train_set':
    tensor_dataset = TensorDataset(torch.from_numpy(X_non_member_train),
                                   torch.from_numpy(y_non_member_train))
elif args.s_test_set == 'member_test_set':
    tensor_dataset = TensorDataset(torch.from_numpy(X_member_test),
                                   torch.from_numpy(y_member_test))
elif args.s_test_set == 'non_member_test_set':
    tensor_dataset = TensorDataset(torch.from_numpy(X_non_member_test),
                                   torch.from_numpy(y_non_member_test))
else:
    raise AssertionError('Unrecognized s_test_set {}'.format(args.s_test_set))
test_loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False,
                         pin_memory=False, drop_last=False)

if args.calc_grad_z:
    logger.info('Start grad_z calculation...')
    calc_grad_z(
        model=net,
        train_loader=train_loader,
        save_pth=os.path.join(OUTPUT_DIR, 'grad_z'),
        gpu=0,
        start=0)

if args.calc_s_test:
    logger.info('Start s_test calculation for {}...'.format(args.s_test_set))
    calc_s_test(
        model=net,
        test_loader=test_loader,
        train_loader=train_loader,
        save=os.path.join(OUTPUT_DIR, 's_test', args.s_test_set),
        gpu=0,
        damp=0.01,
        scale=25,
        recursion_depth=train_loader.dataset.__len__(),
        r=1,
        start=0)

if args.calc_influences:
    logger.info('Start influence scores calculation for {}...'.format(args.s_test_set))
    influences = calc_all_influences(os.path.join(OUTPUT_DIR, 'grad_z'), train_loader.dataset.__len__(),
                                     os.path.join(OUTPUT_DIR, 's_test', args.s_test_set), test_loader.dataset.__len__())
    np.save(os.path.join(OUTPUT_DIR, 'influences', 'influences_' + args.s_test_set + '.npy'), influences)

logger.info('Done.')
