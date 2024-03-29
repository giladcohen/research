'''Attack DNNs with PyTorch.'''
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader

from typing import Callable, Optional, Tuple, Union, Any, List
import numpy as np
import json

from tqdm import tqdm
import os
import argparse
import time
import pickle
import logging
import sys
import random
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path
import matplotlib.pyplot as plt
from cleverhans.utils import random_targets, to_categorical
from robustbench.model_zoo.architectures.dm_wide_resnet import Swish, CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, \
    CIFAR100_STD

sys.path.insert(0, ".")
sys.path.insert(0, "./adversarial_robustness_toolbox")
sys.path.insert(0, "./influence_functions")

from research.losses.losses import L2Loss, LinfLoss, CosineEmbeddingLossV2
from research.datasets.train_val_test_data_loaders import get_test_loader, get_train_valid_loader, \
    get_loader_with_specific_inds, get_normalized_tensor, get_dataset_with_specific_records
from research.datasets.utils import get_robustness_inds
from research.utils import boolean_string, pytorch_evaluate, set_logger, get_image_shape, get_num_classes, \
    get_max_train_size, convert_tensor_to_image, calc_acc_precision_recall, remove_substr_from_keys, calc_auc_roc
from research.models.utils import get_strides, get_conv1_params, get_densenet_conv1_params, get_model

from art.attacks.inference.membership_inference import ShadowModels, LabelOnlyDecisionBoundary, \
    MembershipInferenceBlackBoxRuleBased, MembershipInferenceBlackBox, SelfInfluenceFunctionAttack
from art.estimators.classification import PyTorchClassifier

parser = argparse.ArgumentParser(description='Membership attack script')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/mi/cifar10/resnet18/relu/s_25k_wo_aug', type=str, help='checkpoint dir')
parser.add_argument('--checkpoint_file', default='ckpt.pth', type=str, help='checkpoint path file name')
parser.add_argument('--use_dp_arch', default=False, type=boolean_string, help='Use the same arch used in data privacy training')
parser.add_argument('--attack', default='self_influence', type=str, help='MI attack: gap/black_box/boundary_distance/self_influence')
parser.add_argument('--attacker_knowledge', type=float, default=0.5, help='The portion of samples available to the attacker.')
parser.add_argument('--output_dir', default='self_influence_m_10_nm_100', type=str, help='attack directory')
parser.add_argument('--probabilities', default=True, type=boolean_string, help='Get probabilities for membership')

# member/non-member data config
parser.add_argument('--generate_mi_data', default=False, type=boolean_string, help='To generate MI data')
parser.add_argument('--fast', default=False, type=boolean_string, help='Fast fit (1000 samples) and inference (5000 samples)')
parser.add_argument('--data_dir', default='data', type=str, help='Directory to save the member and non-member training/test data')

# self_influence attack params
parser.add_argument('--miscls_as_nm', default=False, type=boolean_string, help='Label misclassification is inferred as non members')
parser.add_argument('--adaptive', default=False, type=boolean_string, help='Using train loader of influence function with augmentations, adaSIF method')
parser.add_argument('--average', default=False, type=boolean_string, help='Using train loader of influence function with augmentations, ensemble method')
parser.add_argument('--rec_dep', type=int, default=1, help='recursion_depth of the influence functions.')
parser.add_argument('--r', type=int, default=1, help='number of iterations of which to take the avg of the h_estimate calculation.')

# more specific mem/non-mem training/test sizes
parser.add_argument('--n_mem_train', default=None, type=int, help='If not none, use n members for training set.')
parser.add_argument('--n_non_mem_train', default=None, type=int, help='If not none, use n non-members for training set.')
parser.add_argument('--n_mem_test', default=None, type=int, help='If not none, use n members for test set.')
parser.add_argument('--n_non_mem_test', default=None, type=int, help='If not none, use n non-members for test set.')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--host', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

if args.attack == 'boundary_distance':
    assert args.fast or args.n_mem_train, \
        'boundary distance attack is slow and needs to work only with fast=True argument or explicit training/test size'
if args.n_mem_train or args.n_non_mem_train or args.n_mem_test or args.n_non_mem_test:
    assert not args.fast and args.n_mem_train and args.n_non_mem_train and args.n_mem_test and args.n_non_mem_test, \
        "All specific mem/non-mem training and test sizes must be set together."

# for reproduce:
# seed = 9
# torch.manual_seed(seed)
# random.seed(seed)
# np.random.seed(seed)
rand_gen = np.random.RandomState(int(time.time()))

with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, args.checkpoint_file)
if args.output_dir != '':
    OUTPUT_DIR = os.path.join(args.checkpoint_dir, args.output_dir)
else:
    OUTPUT_DIR = os.path.join(args.checkpoint_dir, args.attack)
DATA_DIR = os.path.join(args.checkpoint_dir, args.data_dir)
os.makedirs(os.path.join(OUTPUT_DIR), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR), exist_ok=True)

log_file = os.path.join(OUTPUT_DIR, 'log_auc_opt_no_miscls_as_nm.log')
set_logger(log_file)

# importing opacus just now not to override logger
from opacus.validators import ModuleValidator

logger = logging.getLogger()

dataset = train_args['dataset']
img_shape = get_image_shape(dataset)
num_classes = get_num_classes(dataset)
max_train_size = get_max_train_size(dataset)
batch_size = 100

# Model
logger.info('==> Building model..')
net_cls = get_model(train_args['net'], dataset)
if 'resnet' in train_args['net']:
    conv1 = get_conv1_params(dataset)
    strides = get_strides(dataset)
    net = net_cls(num_classes=num_classes, activation=train_args['activation'], conv1=conv1, strides=strides, field='probs')
elif train_args['net'] == 'alexnet':
    net = net_cls(num_classes=num_classes, activation=train_args['activation'], field='probs')
elif train_args['net'] == 'densenet':
    assert train_args['activation'] == 'relu'
    conv1 = get_densenet_conv1_params(dataset)
    net = net_cls(growth_rate=6, num_layers=52, num_classes=num_classes, drop_rate=0.0, conv1=conv1, field='probs')
else:
    raise AssertionError('Does not support architecture {}'.format(train_args['net']))

if args.use_dp_arch:
    logger.info('Converting to DP architecture')
    net = ModuleValidator.fix(net)

net = net.to(device)
global_state = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))
if 'best_net' in global_state:
    global_state = global_state['best_net']
if '_module' in list(global_state.keys())[0]:
    global_state = remove_substr_from_keys(global_state, '_module.')
net.load_state_dict(global_state, strict=True)
net.eval()
# summary(net, (img_shape[2], img_shape[0], img_shape[1]))
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

optimizer = optim.SGD(
    net.parameters(),
    lr=train_args['lr'],
    momentum=0.9,
    weight_decay=0.0,  # train_args['wd'],
    nesterov=True)
loss = nn.CrossEntropyLoss()
classifier = PyTorchClassifier(model=net, clip_values=(0, 1), loss=loss, optimizer=optimizer,
                               input_shape=(img_shape[2], img_shape[0], img_shape[1]), nb_classes=num_classes)

# Data
if not os.path.exists(os.path.join(DATA_DIR, 'X_member_train.npy')):
    assert args.generate_mi_data, 'MI data does not exist. Generate it by setting generate_mi_data = True'
    logger.info('==> Preparing data..')
    all_train_inds = np.arange(max_train_size)
    train_inds = np.load(os.path.join(args.checkpoint_dir, 'train_inds.npy'))
    val_inds   = np.load(os.path.join(args.checkpoint_dir, 'val_inds.npy'))
    unused_train_inds = np.asarray([i for i in all_train_inds if i not in np.concatenate((train_inds, val_inds))])

    train_loader = get_loader_with_specific_inds(
        dataset=dataset,
        dataset_args=dict(),
        batch_size=batch_size,
        is_training=False,
        indices=train_inds,
        num_workers=0,
        pin_memory=device=='cuda'
    )
    unused_train_loader = get_loader_with_specific_inds(
        dataset=dataset,
        dataset_args=dict(),
        batch_size=batch_size,
        is_training=False,
        indices=unused_train_inds,
        num_workers=0,
        pin_memory=device=='cuda'
    )
    val_loader = get_loader_with_specific_inds(
        dataset=dataset,
        dataset_args=dict(),
        batch_size=batch_size,
        is_training=False,
        indices=val_inds,
        num_workers=0,
        pin_memory=device=='cuda'
    )
    test_loader = get_test_loader(
        dataset=dataset,
        dataset_args=dict(),
        batch_size=batch_size,
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
    # debug
    # X_train_member_img = convert_tensor_to_image(X_train_member)
    # X_train_non_member_img = convert_tensor_to_image(X_train_non_member)
    # X_val_img = convert_tensor_to_image(X_val)
    # X_test_img = convert_tensor_to_image(X_test)

    # combine members and non members and define train/test set
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

def randomize_max_p_points(x: np.ndarray, y: np.ndarray, p: int):
    if x.shape[0] > p:
        logger.info('Selecting {} random rows out of {}'.format(p, x.shape[0]))
        inds = rand_gen.choice(x.shape[0], p, replace=False)
        return x[inds], y[inds]
    else:
        return x, y

if args.fast or args.n_mem_train:
    if args.fast:
        n_mem_train, n_non_mem_train, n_mem_test, n_non_mem_test = 500, 500, 2500, 2500
    else:
        n_mem_train, n_non_mem_train, n_mem_test, n_non_mem_test = args.n_mem_train, args.n_non_mem_train, \
                                                                   args.n_mem_test, args.n_non_mem_test

    if not os.path.exists(os.path.join(OUTPUT_DIR, 'X_member_train_fast.npy')):
        X_member_train, y_member_train = randomize_max_p_points(X_member_train, y_member_train, n_mem_train)
        X_non_member_train, y_non_member_train = randomize_max_p_points(X_non_member_train, y_non_member_train, n_non_mem_train)
        X_member_test, y_member_test = randomize_max_p_points(X_member_test, y_member_test, n_mem_test)
        X_non_member_test, y_non_member_test = randomize_max_p_points(X_non_member_test, y_non_member_test, n_non_mem_test)
        np.save(os.path.join(OUTPUT_DIR, 'X_member_train_fast.npy'), X_member_train)
        np.save(os.path.join(OUTPUT_DIR, 'y_member_train_fast.npy'), y_member_train)
        np.save(os.path.join(OUTPUT_DIR, 'X_non_member_train_fast.npy'), X_non_member_train)
        np.save(os.path.join(OUTPUT_DIR, 'y_non_member_train_fast.npy'), y_non_member_train)
        np.save(os.path.join(OUTPUT_DIR, 'X_member_test_fast.npy'), X_member_test)
        np.save(os.path.join(OUTPUT_DIR, 'y_member_test_fast.npy'), y_member_test)
        np.save(os.path.join(OUTPUT_DIR, 'X_non_member_test_fast.npy'), X_non_member_test)
        np.save(os.path.join(OUTPUT_DIR, 'y_non_member_test_fast.npy'), y_non_member_test)
    else:
        logger.info('loading fast data..')
        X_member_train = np.load(os.path.join(OUTPUT_DIR, 'X_member_train_fast.npy'))
        y_member_train = np.load(os.path.join(OUTPUT_DIR, 'y_member_train_fast.npy'))
        X_non_member_train = np.load(os.path.join(OUTPUT_DIR, 'X_non_member_train_fast.npy'))
        y_non_member_train = np.load(os.path.join(OUTPUT_DIR, 'y_non_member_train_fast.npy'))
        X_member_test = np.load(os.path.join(OUTPUT_DIR, 'X_member_test_fast.npy'))
        y_member_test = np.load(os.path.join(OUTPUT_DIR, 'y_member_test_fast.npy'))
        X_non_member_test = np.load(os.path.join(OUTPUT_DIR, 'X_non_member_test_fast.npy'))
        y_non_member_test = np.load(os.path.join(OUTPUT_DIR, 'y_non_member_test_fast.npy'))

# Rule based attack (aka Gap attack)
logger.info('Fitting {} attack...'.format(args.attack))
start = time.time()
if args.attack == 'gap':
    attack = MembershipInferenceBlackBoxRuleBased(classifier)
elif args.attack == 'black_box':
    attack = MembershipInferenceBlackBox(classifier)
    attack.fit(x=X_member_train, y=y_member_train, test_x=X_non_member_train, test_y=y_non_member_train)
elif args.attack == 'boundary_distance':
    attack = LabelOnlyDecisionBoundary(classifier)
    attack.calibrate_distance_threshold(X_member_train, y_member_train, X_non_member_train, y_non_member_train)
elif args.attack == 'self_influence':
    attack = SelfInfluenceFunctionAttack(classifier, debug_dir=OUTPUT_DIR, miscls_as_nm=args.miscls_as_nm,
                                         adaptive=args.adaptive, average=args.average, for_ref=False,
                                         rec_dep=args.rec_dep, r=args.r, optimize_tpr_fpr=True)
    attack.fit(x_member=X_member_train, y_member=y_member_train,
               x_non_member=X_non_member_train, y_non_member=y_non_member_train)
else:
    err_str = 'Invalid attack {}'.format(args.attack)
    logger.error(err_str)
    raise AssertionError(err_str)
logger.info('Fitting time: {} sec'.format(time.time() - start))

with open(os.path.join(OUTPUT_DIR, 'attack_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

start = time.time()
infer_cfg = {'probabilities': args.probabilities}

if args.attack == 'boundary_distance':
    inferred_member = attack.infer(X_member_test, y_member_test, **infer_cfg)
    inferred_non_member = attack.infer(X_non_member_test, y_non_member_test, **infer_cfg)
else:
    infer_cfg['infer_set'] = 'member_test'
    inferred_member = attack.infer(X_member_test, y_member_test, **infer_cfg)
    infer_cfg['infer_set'] = 'non_member_test'
    inferred_non_member = attack.infer(X_non_member_test, y_non_member_test, **infer_cfg)

if args.probabilities:
    fpr, tpr, thresholds = calc_auc_roc(inferred_non_member, inferred_member)
    np.save(os.path.join(OUTPUT_DIR, 'fpr_opt_no_miscls_as_nm.npy'), fpr)
    np.save(os.path.join(OUTPUT_DIR, 'tpr_opt_no_miscls_as_nm.npy'), tpr)
    np.save(os.path.join(OUTPUT_DIR, 'thresholds_opt_no_miscls_as_nm.npy'), thresholds)
else:
    calc_acc_precision_recall(inferred_non_member, inferred_member)

logger.info('Inference time: {} sec'.format(time.time() - start))
logger.info('done')
logger.handlers[0].flush()


# debug
# self_influences_member = np.load('/data/gilad/logs/mi/cifar10/resnet18/relu/s_25k_wo_aug/self_influence_debug/self_influences_member_train.npy')
# self_influences_non_member = np.load('/data/gilad/logs/mi/cifar10/resnet18/relu/s_25k_wo_aug/self_influence_debug/self_influences_non_member_train.npy')
