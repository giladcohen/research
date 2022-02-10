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
from cleverhans.utils import random_targets, to_categorical

# sys.path.insert(0, ".")
# sys.path.insert(0, "./adversarial_robustness_toolbox")
from research.losses.losses import L2Loss, LinfLoss, CosineEmbeddingLossV2
from research.datasets.train_val_test_data_loaders import get_test_loader, get_train_valid_loader, \
    get_loader_with_specific_inds, get_normalized_tensor
from research.datasets.utils import get_robustness_inds
from research.utils import boolean_string, pytorch_evaluate, set_logger, get_image_shape
from research.models.utils import get_strides, get_conv1_params, get_model
from research.classifiers.pytorch_classifier_specific import PyTorchClassifierSpecific
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, DeepFool, SaliencyMapMethod, \
    CarliniL2Method, CarliniLInfMethod

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 adversarial robustness testing')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/glove_emb/cifar10/resnet34_ref', type=str, help='checkpoint dir')
parser.add_argument('--checkpoint_file', default='ckpt.pth', type=str, help='checkpoint path file name')
parser.add_argument('--attack', default='fgsm', type=str, help='attack: fgsm, jsma, pgd, deepfool, cw')
parser.add_argument('--attack_loss', default='cross_entropy', type=str,
                    help='The loss used for attacking: cross_entropy/L1/SL1/L2/Linf/cosine')
parser.add_argument('--attack_dir', default='debug', type=str, help='attack directory')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--num_workers', default=0, type=int, help='Data loading threads')

# for FGSM/PGD/CW_Linf/whitebox_pgd:
parser.add_argument('--eps'     , default=0.031, type=float, help='maximum Linf deviation from original image')
parser.add_argument('--eps_step', default=0.003, type=float, help='step size of each adv iteration')
parser.add_argument('--max_iter', default=100, type=int, help='Max iter for PGD attack')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

# for reproduce
# torch.manual_seed(9)
# random.seed(9)
# np.random.seed(9)
# rand_gen = np.random.RandomState(seed=12345)

with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)

if args.attack_loss != 'cross_entropy':
    assert (train_args['glove_dim'] is not None) and (train_args['glove_dim'] != -1), 'glove_dim must be > 0'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, args.checkpoint_file)
CLASS_EMB_VECS = os.path.join(args.checkpoint_dir, 'class_emb_vecs.npy')
ATTACK_DIR = os.path.join(args.checkpoint_dir, args.attack_dir)
targeted = args.attack != 'deepfool'
os.makedirs(os.path.join(ATTACK_DIR), exist_ok=True)
batch_size = args.batch_size

log_file = os.path.join(ATTACK_DIR, 'log.log')
set_logger(log_file)
logger = logging.getLogger()

dataset = train_args['dataset']
_, test_inds = get_robustness_inds(dataset)
test_size = len(test_inds)
emb_dim = train_args.get('glove_dim', -1)
dataset_args = {'cls_to_omit': None, 'emb_selection': None, 'emb_dim': emb_dim if emb_dim != -1 else None}

# Data
logger.info('==> Preparing data..')
testloader = get_test_loader(
    dataset=dataset,
    dataset_args=dataset_args,
    batch_size=batch_size,
    num_workers=args.num_workers,
    pin_memory=device=='cuda'
)
img_shape = get_image_shape(dataset)
classes = testloader.dataset.classes
if os.path.exists(CLASS_EMB_VECS):
    testloader.dataset.overwrite_emb_vecs(np.load(CLASS_EMB_VECS))
class_emb_vecs = testloader.dataset.idx_to_class_emb_vec
num_classes = len(classes)

# Model
logger.info('==> Building model..')
conv1 = get_conv1_params(dataset)
strides = get_strides(dataset)
glove_dim = train_args.get('glove_dim', -1)
if glove_dim != -1:
    ext_linear = glove_dim
else:
    ext_linear = None
net = get_model(train_args['net'])(num_classes=num_classes, activation=train_args['activation'],
                                   conv1=conv1, strides=strides, ext_linear=ext_linear)
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

if args.attack_loss == 'cross_entropy':
    loss = nn.CrossEntropyLoss()
elif args.attack_loss == 'L1':
    loss = nn.L1Loss()
elif args.attack_loss == 'SL1':
    loss = nn.SmoothL1Loss()
elif args.attack_loss == 'L2':
    loss = L2Loss()
elif args.attack_loss == 'Linf':
    loss = LinfLoss()
elif args.attack_loss == 'cosine':
    loss = CosineEmbeddingLossV2()
else:
    raise AssertionError('Unknown value args.attack_loss = {}'.format(args.attack_loss))

if args.attack_loss == 'cross_entropy':
    fields = ['logits']
else:
    fields = ['glove_embeddings']

classifier = PyTorchClassifierSpecific(model=net, clip_values=(0, 1), loss=loss,
                                       optimizer=optimizer, input_shape=(img_shape[2], img_shape[0], img_shape[1]),
                                       nb_classes=num_classes, fields=fields)

X_test = get_normalized_tensor(testloader, img_shape, batch_size)
y_test = np.asarray(testloader.dataset.targets)

# attack
# creating targeted labels
if targeted:
    tgt_file = os.path.join(ATTACK_DIR, 'y_test_adv.npy')
    if not os.path.isfile(tgt_file):
        y_test_targets = random_targets(y_test, num_classes)
        y_test_adv = y_test_targets.argmax(axis=1)
        np.save(os.path.join(ATTACK_DIR, 'y_test_adv.npy'), y_test_adv)
    else:
        y_test_adv = np.load(os.path.join(ATTACK_DIR, 'y_test_adv.npy'))

    if args.attack_loss != 'cross_entropy':
        # converting y_test_adv from vector to matrix of embeddings
        y_adv_vec = np.empty((test_size, glove_dim), dtype=np.float32)
        for i in range(test_size):
            y_adv_vec[i] = class_emb_vecs[y_test_adv[i]]
        y_test_adv = y_adv_vec
else:
    y_test_adv = None

if args.attack == 'fgsm':
    attack = FastGradientMethod(
        estimator=classifier,
        norm=np.inf,
        eps=args.eps,
        eps_step=args.eps_step,
        targeted=targeted,
        num_random_init=0,
        batch_size=batch_size
    )
elif args.attack == 'pgd':
    attack = ProjectedGradientDescent(
        estimator=classifier,
        norm=np.inf,
        eps=args.eps,
        eps_step=args.eps_step,
        targeted=targeted,
        num_random_init=10,
        max_iter=args.max_iter,
        batch_size=batch_size
    )
elif args.attack == 'deepfool':
    attack = DeepFool(
        classifier=classifier,
        max_iter=50,
        epsilon=0.02,
        nb_grads=num_classes,
        batch_size=batch_size
    )
elif args.attack == 'jsma':
    attack = SaliencyMapMethod(
        classifier=classifier,
        theta=1.0,
        gamma=0.01,
        batch_size=batch_size
    )
elif args.attack == 'cw':
    attack = CarliniL2Method(
        classifier=classifier,
        confidence=0.8,
        targeted=targeted,
        initial_const=0.1,
        batch_size=batch_size
    )
elif args.attack == 'cw_Linf':
    attack = CarliniLInfMethod(
        classifier=classifier,
        confidence=0.8,
        targeted=targeted,
        batch_size=batch_size,
        eps=args.eps
    )
else:
    raise AssertionError('Invalid args.attack = {}'.format(args.attack))

dump_args = args.__dict__.copy()
dump_args['attack_params'] = {}
for param in attack.attack_params:
    if param in attack.__dict__.keys():
        dump_args['attack_params'][param] = attack.__dict__[param]
with open(os.path.join(ATTACK_DIR, 'attack_args.txt'), 'w') as f:
    json.dump(dump_args, f, indent=2)

if not os.path.exists(os.path.join(ATTACK_DIR, 'X_test_adv.npy')):
    X_test_adv = attack.generate(x=X_test, y=y_test_adv)
    np.save(os.path.join(ATTACK_DIR, 'X_test_adv.npy'), X_test_adv)

logger.handlers[0].flush()
