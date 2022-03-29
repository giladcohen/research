'''Attack DNNs with PyTorch.'''
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchsummary import summary
from torch.utils.data import TensorDataset

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
from pathlib import Path
import matplotlib.pyplot as plt
from cleverhans.utils import random_targets, to_categorical
from robustbench.model_zoo.architectures.dm_wide_resnet import Swish, CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, \
    CIFAR100_STD
from captum.influence import TracInCPFast

# sys.path.insert(0, ".")
# sys.path.insert(0, "./adversarial_robustness_toolbox")
# from research.classifiers.pytorch_classifier_specific import PyTorchClassifierSpecific
from research.losses.losses import L2Loss, LinfLoss, CosineEmbeddingLossV2
from research.datasets.train_val_test_data_loaders import get_test_loader, get_train_valid_loader, \
    get_loader_with_specific_inds, get_normalized_tensor, get_dataset_with_specific_records
from research.datasets.utils import get_robustness_inds
from research.utils import boolean_string, pytorch_evaluate, set_logger, get_image_shape, get_num_classes, \
    get_max_train_size, convert_tensor_to_image, calc_acc_precision_recall
from research.models.utils import get_strides, get_conv1_params, get_model

from art.attacks.inference.membership_inference import ShadowModels, LabelOnlyDecisionBoundary, \
    MembershipInferenceBlackBoxRuleBased, MembershipInferenceBlackBox, TracInAttack
from art.estimators.classification import PyTorchClassifier
from influence_functions.pytorch_influence_functions.influence_functions.influence_functions import load_grad_z, \
    load_s_test
from influence_functions.pytorch_influence_functions import display_progress

parser = argparse.ArgumentParser(description='Membership attack script')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/mi/cifar10/resnet18/s_1k_wo_aug_act_swish', type=str, help='checkpoint dir')
parser.add_argument('--checkpoint_file', default='ckpt.pth', type=str, help='checkpoint path file name')
# parser.add_argument('--attack', default='shadow_models', type=str, help='attack: shadow_models')
parser.add_argument('--attacker_knowledge', type=float, default=0.5,
                    help='The portion of samples available to the attacker.')
parser.add_argument('--output_dir', default='influence_functions', type=str, help='attack directory')
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
OUTPUT_DIR = os.path.join(args.checkpoint_dir, args.output_dir)
DATA_DIR = os.path.join(args.checkpoint_dir, 'data')
os.makedirs(os.path.join(OUTPUT_DIR), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR), exist_ok=True)

log_file = os.path.join(OUTPUT_DIR, 'log.log')
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
if not os.path.exists(os.path.join(DATA_DIR, 'X_member_train.npy')):
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

# # Rule based attack (aka Gap attack)
# logger.info('Running Rule Based attack...')
# attack = MembershipInferenceBlackBoxRuleBased(classifier)
# inferred_member = attack.infer(X_member_test, y_member_test)
# inferred_non_member = attack.infer(X_non_member_test, y_non_member_test)
# calc_acc_precision_recall(inferred_non_member, inferred_member)
#
# # Black-box attack
# logger.info('Running black box attack...')
# attack = MembershipInferenceBlackBox(classifier)
# attack.fit(x=X_member_train, y=y_member_train, test_x=X_non_member_train, test_y=y_non_member_train)
# inferred_member = attack.infer(X_member_test, y_member_test)
# inferred_non_member = attack.infer(X_non_member_test, y_non_member_test)
# calc_acc_precision_recall(inferred_non_member, inferred_member)

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
# logger.info('Running Boundary distance attack...')
# attack = LabelOnlyDecisionBoundary(classifier)
# # attack.distance_threshold_tau = 0.357647066116333
# attack.calibrate_distance_threshold(x_train=X_member_train, y_train=y_member_train,
#                                     x_test=X_non_member_train, y_test=y_non_member_train)
# inferred_member = attack.infer(X_member_test, y_member_test)
# inferred_non_member = attack.infer(X_non_member_test, y_non_member_test)
# calc_acc_precision_recall(inferred_non_member, inferred_member)
# # 03/14/2022 02:48:28 AM root INFO member acc: 0.988, non-member acc: 0.866, balanced acc: 0.927, precision/recall(member): 0.8805704099821747/0.988, precision/recall(non-member): 0.9863325740318907/0.866

# # TracIn
# logger.info('Running TracIn attack...')
# # self influence method
# # tracin = TracInCPFast(
# #     model=net,
# #     final_fc_layer=net.linear,
# #     influence_src_dataset=member_src_set,
# #     checkpoints_load_func=load_state_dict,
# #     checkpoints=[CHECKPOINT_PATH],
# #     loss_fn=nn.CrossEntropyLoss(reduction="sum"),
# #     batch_size=membership_train_size,
# #     vectorize=False,
# # )
# # member_self_influence_scores = tracin.influence(
# #     inputs=None,
# #     targets=None,
# # )
# #
# # tracin = TracInCPFast(
# #     model=net,
# #     final_fc_layer=net.linear,
# #     influence_src_dataset=non_member_src_set,
# #     checkpoints_load_func=load_state_dict,
# #     checkpoints=[CHECKPOINT_PATH],
# #     loss_fn=nn.CrossEntropyLoss(reduction="sum"),
# #     batch_size=non_membership_train_size,
# #     vectorize=False,
# # )
# # non_member_self_influence_scores = tracin.influence(
# #     inputs=None,
# #     targets=None,
# # )
#
# # use the fact that we know some on the training set of the model
# attack = TracInAttack(classifier)
# attack.fit(X_member_train, y_member_train, X_non_member_train, y_non_member_train, CHECKPOINT_PATH)
# inferred_member = attack.infer(X_member_test, y_member_test)
# inferred_non_member = attack.infer(X_non_member_test, y_non_member_test)
# calc_acc_precision_recall(inferred_non_member, inferred_member)
# # 03/21/2022 05:08:32 PM root INFO member acc: 0.942, non-member acc: 0.722, balanced acc: 0.832, precision/recall(member): 0.7721311475409836/0.942, precision/recall(non-member): 0.9256410256410257/0.722

# Influence functions
logger.info('Running Influence Functions attack...')
# member_src_set = TensorDataset(torch.from_numpy(X_member_train).to(device), torch.from_numpy(y_member_train))
# non_member_src_set = TensorDataset(torch.from_numpy(X_non_member_train).to(device), torch.from_numpy(y_non_member_train))

train_size = X_member_train.shape[0]
grad_z_vecs = load_grad_z(grad_z_dir=os.path.join(OUTPUT_DIR, 'grad_z'), train_dataset_size=train_size)
s_test_dir = os.path.join(OUTPUT_DIR, 's_test', 'member_train_set')
num_s_test_files = len(list(Path(s_test_dir).glob("*.s_test")))
suffix = 'recdep500_r1'
s_test_vecs = []
for i in range(num_s_test_files):
    # s_test.append(torch.load(s_test_dir / str(s_test_id) + f"_{i}.s_test"))
    s_test_vecs.append(torch.load(os.path.join(s_test_dir, str(i) + '_' + suffix + '.s_test')))
    display_progress("s_test files loaded: ", i, num_s_test_files)

influences = torch.zeros(num_s_test_files, train_size)
for i in tqdm(range(num_s_test_files)):
    s_test_vec = s_test_vecs[i]
    for j in range(train_size):
        grad_z_vec = grad_z_vecs[j]
        with torch.no_grad():
            tmp_influence = (
                    -sum(
                        [
                            ####################
                            # TODO: potential bottle neck, takes 17% execution time
                            # torch.sum(k * j).data.cpu().numpy()
                            ####################
                            torch.sum(k * j).data
                            for k, j in zip(grad_z_vec, s_test_vec)
                        ]
                    )
                    / train_size
            )
        influences[i, j] = tmp_influence





# e_s_test, _ = load_s_test(train_dataset_size=train_dataset_size)
# e_s_test, _ = load_s_test(s_test_dir=os.path.join(OUTPUT_DIR, 's_test', 'member_train_set'),
#                           r_sample_size=X_member_train.shape[0],
#                           train_dataset_size=train_size,
#                           suffix='recdep{}_r1'.format(train_size))








with open(os.path.join(OUTPUT_DIR, 'attack_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)


logger.handlers[0].flush()


