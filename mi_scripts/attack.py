'''Attack DNNs with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchsummary import summary

import numpy as np
import json
import os
import argparse
import time
import pickle
import logging
import sys
import random
import matplotlib.pyplot as plt
from cleverhans.utils import random_targets, to_categorical
from robustbench.model_zoo.architectures.dm_wide_resnet import Swish, CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, \
    CIFAR100_STD

# sys.path.insert(0, ".")
# sys.path.insert(0, "./adversarial_robustness_toolbox")
# from research.classifiers.pytorch_classifier_specific import PyTorchClassifierSpecific
from research.losses.losses import L2Loss, LinfLoss, CosineEmbeddingLossV2
from research.datasets.train_val_test_data_loaders import get_test_loader, get_train_valid_loader, \
    get_loader_with_specific_inds, get_normalized_tensor
from research.datasets.utils import get_robustness_inds
from research.utils import boolean_string, pytorch_evaluate, set_logger, get_image_shape, get_num_classes, \
    get_max_train_size, convert_tensor_to_image, calc_acc_precision_recall
from research.models.utils import get_strides, get_conv1_params, get_model

from art.attacks.inference.membership_inference import ShadowModels, LabelOnlyDecisionBoundary, \
    MembershipInferenceBlackBoxRuleBased, MembershipInferenceBlackBox

from art.estimators.classification import PyTorchClassifier

parser = argparse.ArgumentParser(description='Membership attack script')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/mi/cifar10/resnet18/s_1k_wo_aug_act_swish', type=str, help='checkpoint dir')
parser.add_argument('--checkpoint_file', default='ckpt.pth', type=str, help='checkpoint path file name')
parser.add_argument('--attack', default='shadow_models', type=str, help='attack: shadow_models')
parser.add_argument('--attacker_knowledge', type=float, default=0.5,
                    help='The portion of samples available to the attacker.')
parser.add_argument('--attack_dir', default='shadow_models', type=str, help='attack directory')
parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

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
ATTACK_DIR = os.path.join(args.checkpoint_dir, args.attack_dir)
os.makedirs(os.path.join(ATTACK_DIR), exist_ok=True)

log_file = os.path.join(ATTACK_DIR, 'log.log')
set_logger(log_file)
logger = logging.getLogger()

dataset = train_args['dataset']
img_shape = get_image_shape(dataset)
num_classes = get_num_classes(dataset)
max_train_size = get_max_train_size(dataset)
batch_size = 100

# Model
logger.info('==> Building model..')
net_cls = get_model(train_args['net'])
if 'resnet' in train_args['net']:
    conv1 = get_conv1_params(dataset)
    strides = get_strides(dataset)
    net = net_cls(num_classes=num_classes, activation=train_args['activation'], conv1=conv1, strides=strides, field='probs')
else:
    raise AssertionError('Does not support non Resnet architectures')
net = net.to(device)
global_state = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))
if 'best_net' in global_state:
    global_state = global_state['best_net']
net.load_state_dict(global_state)
net.eval()
# summary(net, (img_shape[2], img_shape[0], img_shape[1]))
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

optimizer = optim.SGD(
    net.parameters(),
    lr=train_args['lr'],
    momentum=train_args['mom'],
    weight_decay=0.0,  # train_args['wd'],
    nesterov=train_args['mom'] > 0)
loss = nn.CrossEntropyLoss()
classifier = PyTorchClassifier(model=net, clip_values=(0, 1), loss=loss, optimizer=optimizer,
                               input_shape=(img_shape[2], img_shape[0], img_shape[1]), nb_classes=num_classes)

# Data
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

# building train/test set for non members
non_membership_test_size = membership_test_size  # same as the membership test size
non_membership_train_size = X_non_member.shape[0] - non_membership_test_size  # as much as possible (can be used by shadow models)
train_non_member_inds = rand_gen.choice(X_non_member.shape[0], non_membership_train_size, replace=False)
train_non_member_inds.sort()
X_non_member_train = X_non_member[train_non_member_inds]  # cam be used by shadow models
y_non_member_train = y_non_member[train_non_member_inds]

test_non_member_inds = np.asarray([i for i in np.arange(X_non_member.shape[0]) if i not in train_non_member_inds])
test_non_member_inds = rand_gen.choice(test_non_member_inds, non_membership_test_size, replace=False)
test_non_member_inds.sort()
X_non_member_test = X_non_member[test_non_member_inds]
y_non_member_test = y_non_member[test_non_member_inds]

assert X_member_test.shape[0] == X_non_member_test.shape[0], 'assert balanced test set for member/non-member'

# Rule based attack (aka Gap attack)
# logger.info('Running Rule Based attack...')
# attack = MembershipInferenceBlackBoxRuleBased(classifier)
# inferred_member = attack.infer(X_member_test, y_member_test)
# inferred_non_member = attack.infer(X_non_member_test, y_non_member_test)
# calc_acc_precision_recall(inferred_non_member, inferred_member)

# Black-box attack
logger.info('Running black box attack...')
attack = MembershipInferenceBlackBox(classifier)
attack.fit(x=X_member_train, y=y_member_train, test_x=X_non_member_train, test_y=y_non_member_train)
inferred_member = attack.infer(X_member_test, y_member_test)
inferred_non_member = attack.infer(X_non_member_test, y_non_member_test)
calc_acc_precision_recall(inferred_non_member, inferred_member)

# shadow models attack
# num_shadow_models = np.min([3, (X_non_member.shape[0] - non_membership_test_size) // (membership_train_size + membership_test_size)])
# shadow_size = num_shadow_models * (membership_train_size + membership_test_size)
# if num_shadow_models > 0:
#     # building shadow dataset (non-member and not in test samples)
#     logger.info('Running shadow models attack...')
#     shadow_non_member_inds = np.asarray([i for i in np.arange(X_non_member.shape[0]) if i not in test_non_member_inds])
#     shadow_non_member_inds = rand_gen.choice(shadow_non_member_inds, shadow_size, replace=False)
#     shadow_non_member_inds.sort()
#     X_shadow = X_non_member[shadow_non_member_inds]
#     y_shadow = y_non_member[shadow_non_member_inds]
#
#     # building the shadow models
#     shadow_models = ShadowModels(classifier, num_shadow_models=num_shadow_models)
#     shadow_dataset = shadow_models.generate_shadow_dataset(X_shadow, to_categorical(y_shadow, num_classes))
#     (member_x, member_y, member_predictions), (nonmember_x, nonmember_y, nonmember_predictions) = shadow_dataset
#     # Shadow models' accuracy
#     print([np.mean(np.argmax(sm.predict(X_test), axis=1) == y_test) for sm in shadow_models.get_shadow_models()])
#
#     attack = MembershipInferenceBlackBox(classifier)
#     attack.fit(member_x, member_y, nonmember_x, nonmember_y, member_predictions, nonmember_predictions)
#     inferred_member = attack.infer(X_member_test, y_member_test)
#     inferred_non_member = attack.infer(X_non_member_test, y_non_member_test)
#     calc_acc_precision_recall(inferred_non_member, inferred_member)

# Boundary distance
logger.info('Running Boundary distance attack...')
attack = LabelOnlyDecisionBoundary(classifier)
# attack.distance_threshold_tau = 0.357647066116333
attack.calibrate_distance_threshold(x_train=X_member_train, y_train=y_member_train,
                                    x_test=X_non_member_train, y_test=y_non_member_train)
inferred_member = attack.infer(X_member_test, y_member_test)
inferred_non_member = attack.infer(X_non_member_test, y_non_member_test)
calc_acc_precision_recall(inferred_non_member, inferred_member)
# 03/14/2022 02:48:28 AM root INFO member acc: 0.988, non-member acc: 0.866, balanced acc: 0.927, precision/recall(member): 0.8805704099821747/0.988, precision/recall(non-member): 0.9863325740318907/0.866




with open(os.path.join(ATTACK_DIR, 'attack_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)


logger.handlers[0].flush()
