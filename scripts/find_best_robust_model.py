'''Attack DNNs with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from sklearn.neighbors import NearestNeighbors
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

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
import glob
from cleverhans.utils import random_targets, to_categorical
from robustbench.model_zoo.architectures.dm_wide_resnet import Swish, CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, \
    CIFAR100_STD

# sys.path.insert(0, ".")
# sys.path.insert(0, "./adversarial_robustness_toolbox")
from research.losses.losses import L2Loss, LinfLoss, CosineEmbeddingLossV2
from research.datasets.train_val_test_data_loaders import get_test_loader, get_train_valid_loader, \
    get_loader_with_specific_inds, get_normalized_tensor
from research.datasets.utils import get_robustness_inds
from research.utils import boolean_string, pytorch_evaluate, set_logger, get_image_shape, convert_tensor_to_image
from research.models.utils import get_strides, get_conv1_params, get_model
from research.classifiers.pytorch_classifier_specific import PyTorchClassifierSpecific
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, DeepFool, SaliencyMapMethod, \
    CarliniL2Method, CarliniLInfMethod, AutoProjectedGradientDescent

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 adversarial robustness testing')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/glove_emb/cifar10/resnet18/baseline1', type=str, help='checkpoint dir')
parser.add_argument('--attack_loss', default='cross_entropy', type=str,
                    help='The loss used for attacking: cross_entropy/L1/L2/Linf/cosine')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--eps'     , default=0.031, type=float, help='maximum Linf deviation from original image')
parser.add_argument('--eps_step'     , default=0.00443, type=float, help='eps step size for adversarial image attack')
parser.add_argument('--steps', default=7, type=int, help='Number of iterations for attack')

# evaluation
parser.add_argument('--method', default='softmax', type=str, help='softmax/knn/cosine')
parser.add_argument('--knn_norm', default="2", type=str, help='Norm for knn: 1/2/inf')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

# for reproduce
# torch.manual_seed(9)
# random.seed(9)
# np.random.seed(9)
# rand_gen = np.random.RandomState(seed=12345)

if args.eps > 1.0:
    args.eps /= 255
if args.eps_step > 1.0:
    args.eps_step /= 255

with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)

if args.attack_loss != 'cross_entropy':
    assert (train_args['glove_dim'] is not None) and (train_args['glove_dim'] != -1), 'glove_dim must be > 0'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASS_EMB_VECS = os.path.join(args.checkpoint_dir, 'class_emb_vecs.npy')
ATTACK_DIR = os.path.join(args.checkpoint_dir, 'ifgsm')
os.makedirs(os.path.join(ATTACK_DIR), exist_ok=True)
batch_size = args.batch_size

log_file = os.path.join(ATTACK_DIR, 'log.log')
set_logger(log_file)
logger = logging.getLogger()
writer = SummaryWriter(os.path.join(ATTACK_DIR, 'performance_vs_epoch'))
####################################### Get validation set #######################################
dataset = train_args['dataset']
val_inds = np.load(os.path.join(args.checkpoint_dir, 'val_inds.npy'))
val_size = len(val_inds)
emb_dim = train_args['glove_dim']
dataset_args = {'cls_to_omit': None, 'emb_selection': None, 'emb_dim': emb_dim if emb_dim != -1 else None}

logger.info('==> Preparing data..')
valloader = get_loader_with_specific_inds(
    dataset,
    dataset_args,
    batch_size,
    is_training=False,
    indices=val_inds,
    num_workers=0,
    pin_memory=False)

img_shape = get_image_shape(dataset)
classes = valloader.dataset.classes
if os.path.exists(CLASS_EMB_VECS):
    logger.info('Loading embeddings vecs from {}'.format(CLASS_EMB_VECS))
    valloader.dataset.overwrite_emb_vecs(np.load(CLASS_EMB_VECS))
    class_emb_vecs = valloader.dataset.idx_to_class_emb_vec
num_classes = len(classes)
X_val = get_normalized_tensor(valloader, img_shape, batch_size)
y_val = np.asarray(valloader.dataset.targets)
##################################################################################################
####################################### Set network        #######################################
# Model
logger.info('==> Building model..')
net_cls = get_model(train_args['net'])
ext_linear = emb_dim if emb_dim != -1 else None
if 'resnet' in train_args['net']:
    conv1 = get_conv1_params(dataset)
    strides = get_strides(dataset)
    net = net_cls(num_classes=num_classes, activation=train_args['activation'], conv1=conv1,
                  strides=strides, ext_linear=ext_linear)
else:
    net = net_cls(num_classes=num_classes, depth=28, width=10, activation_fn=Swish,
                  mean=CIFAR10_MEAN, std=CIFAR10_STD, ext_linear=ext_linear)
net = net.to(device)
net.eval()
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
##################################################################################################
####################################### Set opt and classifier ###################################
optimizer = optim.SGD(
    net.parameters(),
    lr=train_args['lr'],
    momentum=train_args['mom'],
    weight_decay=0.0,
    nesterov=train_args['mom'] > 0)

if args.attack_loss == 'cross_entropy':
    loss = nn.CrossEntropyLoss()
elif args.attack_loss == 'L1':
    loss = nn.L1Loss()
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
##################################################################################################
####################################### Set IFGSM attack #########################################
attack = ProjectedGradientDescent(
    estimator=classifier,
    norm=np.inf,
    eps=args.eps,
    eps_step=args.eps_step,
    targeted=False,
    num_random_init=0,
    max_iter=args.steps,
    batch_size=batch_size
)
dump_args = args.__dict__.copy()
dump_args['attack_params'] = {}
for param in attack.attack_params:
    if param in attack.__dict__.keys():
        dump_args['attack_params'][param] = attack.__dict__[param]
with open(os.path.join(ATTACK_DIR, 'attack_args.txt'), 'w') as f:
    json.dump(dump_args, f, indent=2)
##################################################################################################
################################# Set prediction function ########################################
if args.method == 'knn':
    knn = NearestNeighbors(n_neighbors=1, algorithm='brute', p=args.knn_norm)
    knn.fit(class_emb_vecs)
elif args.method == 'cosine':
    cos = nn.CosineSimilarity()

def pred(X):
    if args.method == 'softmax':
        y_probs = pytorch_evaluate(net, X, ['probs'], batch_size)[0]
        y_preds = y_probs.argmax(axis=1)
    elif args.method == 'knn':
        glove_embs = pytorch_evaluate(net, X, ['glove_embeddings'], batch_size)[0]
        y_preds = knn.kneighbors(glove_embs, return_distance=False).squeeze()
    elif args.method == 'cosine':
        glove_embs = pytorch_evaluate(net, X, ['glove_embeddings'], batch_size, to_tensor=True)[0]
        distance_mat = torch.zeros((val_size, num_classes)).to(device)
        for cls_idx in range(num_classes):
            embs = np.tile(class_emb_vecs[cls_idx], (val_size, 1))
            embs = torch.from_numpy(embs).to(device)
            distance_mat[:, cls_idx] = cos(glove_embs, embs)
        distance_mat = distance_mat.cpu().numpy()
        y_preds = distance_mat.argmax(1)
    else:
        raise AssertionError('Unsupported eval method {}'.format(args.method))
    return y_preds
##################################################################################################
################################# Attacking and evaluating checkpoints ###########################
# files = glob.glob(os.path.join(args.checkpoint_dir, '*.pth'), recursive=False)
# files.sort()
if args.attack_loss == 'cross_entropy':
    targets = y_val
else:
    # converting class labels from vector to matrix of embeddings
    targets = np.empty((val_size, emb_dim), dtype=np.float32)
    for i in range(val_size):
        targets[i] = class_emb_vecs[y_val[i]]

accuracy_dict = dict()
ckpt_list = [os.path.join(args.checkpoint_dir, 'ckpt.pth')]
for epoch in range(1, 400):
    ckpt = os.path.join(args.checkpoint_dir, 'ckpt_epoch_{}.pth'.format(epoch))
    ckpt_list.append(ckpt)
for ckpt in ckpt_list:
    global_state = torch.load(ckpt, map_location=torch.device(device))
    if 'best_net' in global_state:
        global_state = global_state['best_net']
    net.load_state_dict(global_state)
    y_preds = pred(X_val)
    normal_acc = np.mean(y_val == y_preds)

    X_val_adv = attack.generate(x=X_val, y=targets)
    y_val_preds = pred(X_val_adv)
    robust_acc = np.mean(y_val == y_val_preds)

    if os.path.basename(ckpt) != 'ckpt.pth':
        epoch = int(ckpt.split('_')[-1].split('.')[0])
        writer.add_scalar('Accuracy/Clean Validation Acc', normal_acc, epoch)
        writer.add_scalar('Accuracy/IFGSM-7 Validation Acc', robust_acc, epoch)
    else:
        epoch = 'best'
    accuracy_dict[epoch] = {'normal': normal_acc, 'robust': robust_acc}
    logger.info('Epoch {}: normal accuracy: {}%, robust accuracy: {}%'
                .format(epoch, 100.0 * normal_acc, 100.0 * robust_acc))

    writer.flush()
    logger.handlers[0].flush()
