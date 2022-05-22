'''Attack DNNs with PyTorch.'''
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchsummary import summary
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader

from typing import Callable, Optional, Tuple, Union, Any, List
import numpy as np
import json
from collections import OrderedDict
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

# from research.classifiers.pytorch_classifier_specific import PyTorchClassifierSpecific
from research.losses.losses import L2Loss, LinfLoss, CosineEmbeddingLossV2
from research.datasets.train_val_test_data_loaders import get_test_loader, get_train_valid_loader, \
    get_loader_with_specific_inds, get_normalized_tensor, get_dataset_with_specific_records
from research.datasets.utils import get_robustness_inds
from research.utils import boolean_string, pytorch_evaluate, set_logger, get_image_shape, get_num_classes, \
    get_max_train_size, convert_tensor_to_image, calc_acc_precision_recall
from research.models import AlexNetRef, ResNet110Ref, DenseNetRef
from research.models.utils import get_strides, get_conv1_params, get_densenet_conv1_params, get_model

from art.attacks.inference.membership_inference import ShadowModels, LabelOnlyDecisionBoundary, \
    MembershipInferenceBlackBoxRuleBased, MembershipInferenceBlackBox, SelfInfluenceFunctionAttack, \
    InfluenceFunctionDiff
from art.estimators.classification import PyTorchClassifier

parser = argparse.ArgumentParser(description='Membership attack script')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/mi/cifar100/alexnet_ref', type=str, help='checkpoint dir')
parser.add_argument('--checkpoint_file', default='ckpt.pth', type=str, help='checkpoint path file name')
parser.add_argument('--arch', default='alexnet', type=str, help='can be alexnet/resnet/densenet')
parser.add_argument('--attack', default='self_influence', type=str, help='MI attack: gap/black_box/boundary_distance/self_influence')
# parser.add_argument('--attacker_knowledge', type=float, default=0.5, help='The portion of samples available to the attacker.')
parser.add_argument('--output_dir', default='adaptive/self_influence_v3', type=str, help='attack directory')
parser.add_argument('--generate_mi_data', default=False, type=boolean_string, help='To generate MI data')
parser.add_argument('--fast', default=True, type=boolean_string, help='Fast fit (500 samples) and inference (2500 samples)')

# self_influence attack params
parser.add_argument('--miscls_as_nm', default=True, type=boolean_string, help='Label misclassification is inferred as non members')
parser.add_argument('--adaptive', default=True, type=boolean_string, help='Using train loader of influence function with augmentations')
parser.add_argument('--rec_dep', type=int, default=8, help='recursion_depth of the influence functions.')
parser.add_argument('--r', type=int, default=8, help='number of iterations of which to take the avg of the h_estimate calculation.')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

if args.attack == 'boundary_distance':
    assert args.fast, 'boundary distance attack is slow and needs to work only with fast=True argument'

# for reproduce:
# seed = 9
# torch.manual_seed(seed)
# random.seed(seed)
# np.random.seed(seed)
rand_gen = np.random.RandomState(int(time.time()))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, args.checkpoint_file)
if args.output_dir != '':
    OUTPUT_DIR = os.path.join(args.checkpoint_dir, args.output_dir)
else:
    OUTPUT_DIR = os.path.join(args.checkpoint_dir, args.attack)
DATA_DIR = os.path.join(args.checkpoint_dir, 'data')
os.makedirs(os.path.join(OUTPUT_DIR), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR), exist_ok=True)

log_file = os.path.join(OUTPUT_DIR, 'log.log')
set_logger(log_file)
logger = logging.getLogger()

dataset = 'cifar100'
img_shape = get_image_shape(dataset)
num_classes = get_num_classes(dataset)
max_train_size = get_max_train_size(dataset)
batch_size = 100

# Model
logger.info('==> Building model..')
if args.arch == 'alexnet':
    net = AlexNetRef()
elif args.arch == 'resnet':
    net = ResNet110Ref()
elif args.arch == 'densenet':
    net = DenseNetRef()
else:
    raise AssertionError('Must provide architecture name. args.arch={}'.format(args.arch))

net = net.to(device)
global_state = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))
logger.info('The best accuracy for arch {} is {}'.format(args.arch, global_state['best_acc']))
global_state = global_state['state_dict']
global_state = OrderedDict((key.split('module.')[1], value) for (key, value) in global_state.items())
net.load_state_dict(global_state)
net.eval()
# summary(net, (img_shape[2], img_shape[0], img_shape[1]))
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

optimizer = optim.SGD(
    net.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=0.0,
    nesterov=True)
loss = nn.CrossEntropyLoss()
classifier = PyTorchClassifier(model=net, clip_values=(0, 1), loss=loss, optimizer=optimizer,
                               input_shape=(img_shape[2], img_shape[0], img_shape[1]), nb_classes=num_classes)

# Data
if not os.path.exists(os.path.join(DATA_DIR, 'X_member_train.npy')):
    assert args.generate_mi_data, 'MI data does not exist. Generate it by setting generate_mi_data = True'
    logger.info('==> Preparing data..')
    train_inds = np.arange(50000)

    train_loader = get_loader_with_specific_inds(
        dataset=dataset,
        dataset_args=dict(),
        batch_size=batch_size,
        is_training=False,
        indices=train_inds,
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
    X_test = get_normalized_tensor(test_loader, img_shape, batch_size)
    y_test = np.asarray(test_loader.dataset.targets)

    # debug
    # X_train_member_img = convert_tensor_to_image(X_train_member)
    # X_train_non_member_img = convert_tensor_to_image(X_train_non_member)
    # X_val_img = convert_tensor_to_image(X_val)
    # X_test_img = convert_tensor_to_image(X_test)

    # combine members and non members and define train/test set
    X_member = X_train_member
    y_member = y_train_member
    X_non_member = X_test
    y_non_member = y_test
    # building new training and test set
    # building train/test set for members
    membership_train_size = 5000
    membership_test_size = 5000
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

if args.fast:
    # to reproduce, we collect the same samples that were selected from a previous "fast" run, it they exist
    if not os.path.exists(os.path.join(OUTPUT_DIR, 'X_member_train_fast.npy')):
        X_member_train, y_member_train = randomize_max_p_points(X_member_train, y_member_train, 500)  # reduced to 50 to save time
        X_non_member_train, y_non_member_train = randomize_max_p_points(X_non_member_train, y_non_member_train, 500)
        X_member_test, y_member_test = randomize_max_p_points(X_member_test, y_member_test, 2500)
        X_non_member_test, y_non_member_test = randomize_max_p_points(X_non_member_test, y_non_member_test, 2500)
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
                                         adaptive_for_ref=args.adaptive, rec_dep=args.rec_dep, r=args.r)
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

if args.attack == 'boundary_distance':
    inferred_member = attack.infer(X_member_test, y_member_test)
    inferred_non_member = attack.infer(X_non_member_test, y_non_member_test)
else:
    inferred_member = attack.infer(X_member_test, y_member_test, **{'infer_set': 'member_test'})
    inferred_non_member = attack.infer(X_non_member_test, y_non_member_test, **{'infer_set': 'non_member_test'})
calc_acc_precision_recall(inferred_non_member, inferred_member)
logger.info('Inference time: {} sec'.format(time.time() - start))
logger.info('done')
logger.handlers[0].flush()


# debug
# self_influences_member = np.load('/data/gilad/logs/mi/cifar10/resnet18/relu/s_25k_wo_aug/self_influence_debug/self_influences_member_train.npy')
# self_influences_non_member = np.load('/data/gilad/logs/mi/cifar10/resnet18/relu/s_25k_wo_aug/self_influence_debug/self_influences_non_member_train.npy')
