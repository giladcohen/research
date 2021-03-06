'''Train DNNs with GloVe via PyTorch.'''
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchsummary import summary
from torchvision import transforms
from typing import Tuple, Any, Dict
import copy
import numpy as np
import json
import os
import argparse
from tqdm import tqdm
import time
import sys
import logging

from research.datasets.train_val_test_data_loaders import get_test_loader, get_train_valid_loader, \
    get_loader_with_specific_inds
from research.utils import boolean_string, get_image_shape, set_logger, get_parameter_groups, force_lr
from research.models.utils import get_strides, get_conv1_params, get_densenet_conv1_params, get_model

parser = argparse.ArgumentParser(description='Training networks using PyTorch')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/mi/debug13', type=str, help='checkpoint dir')

# dataset
parser.add_argument('--dataset', default='cifar100', type=str, help='dataset: cifar10, cifar100, svhn, tiny_imagenet')
parser.add_argument('--train_size', default=1.0, type=float, help='Fraction of train size out of entire trainset')
parser.add_argument('--augmentations', default=False, type=boolean_string, help='whether to include data augmentations')

# architecture:
parser.add_argument('--net', default='resnet18', type=str, help='network architecture')
parser.add_argument('--activation', default='relu', type=str, help='network activation: relu, softplus, or swish')

# optimization:
parser.add_argument('--resume', default=None, type=str, help='Path to checkpoint to be resumed')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--mom', default=0.9, type=float, help='weight momentum of SGD optimizer')
parser.add_argument('--epochs', default='300', type=int, help='number of epochs')
parser.add_argument('--wd', default=0.0005, type=float, help='weight decay')  # was 5e-4 for batch_size=128
parser.add_argument('--num_workers', default=8, type=int, help='Data loading threads')
parser.add_argument('--metric', default='accuracy', type=str, help='metric to optimize. accuracy or sparsity')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--test_batch_size', default=100, type=int, help='batch size')

# LR scheduler
parser.add_argument('--lr_scheduler', default='multi_step', type=str, help='reduce_on_plateau/multi_step')
parser.add_argument('--factor', default=0.9, type=float, help='LR schedule factor')
parser.add_argument('--patience', default=3, type=int, help='LR schedule patience')
parser.add_argument('--cooldown', default=0, type=int, help='LR cooldown')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

torch.autograd.set_detect_anomaly(True)

# dumping args to txt file
os.makedirs(args.checkpoint_dir, exist_ok=True)
with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')
log_file = os.path.join(args.checkpoint_dir, 'log.log')
batch_size = args.batch_size
test_batch_size = args.test_batch_size

set_logger(log_file)
logger = logging.getLogger()
if args.metric == 'accuracy':
    WORST_METRIC = 0.0
    metric_mode = 'max'
elif args.metric == 'loss':
    WORST_METRIC = np.inf
    metric_mode = 'min'
else:
    raise AssertionError('illegal argument metric={}'.format(args.metric))

rand_gen = np.random.RandomState(int(time.time()))  # we want different nets for ensemble, for reproducibility one
# might want to replace the time with a contant.
train_writer = SummaryWriter(os.path.join(args.checkpoint_dir, 'train'))
val_writer   = SummaryWriter(os.path.join(args.checkpoint_dir, 'val'))
test_writer  = SummaryWriter(os.path.join(args.checkpoint_dir, 'test'))

# Data
logger.info('==> Preparing data..')
dataset_args = dict()
if not args.augmentations:
    dataset_args['train_transform'] = transforms.ToTensor()

train_inds = np.arange(50000)
val_inds = []
trainloader = get_loader_with_specific_inds(
    dataset=args.dataset,
    dataset_args=dataset_args,
    batch_size=batch_size,
    is_training=True,
    indices=train_inds,
    num_workers=args.num_workers,
    pin_memory=device=='cuda'
)
testloader = get_test_loader(
    dataset=args.dataset,
    dataset_args=dataset_args,
    batch_size=test_batch_size,
    num_workers=args.num_workers,
    pin_memory=device=='cuda'
)

img_shape = get_image_shape(args.dataset)
classes = trainloader.dataset.classes
num_classes = len(classes)
train_size = len(trainloader.dataset)
test_size  = len(testloader.dataset)

y_train    = np.asarray(trainloader.dataset.targets)
y_val      = np.asarray([])
y_test     = np.asarray(testloader.dataset.targets)

# dump to dir:
np.save(os.path.join(args.checkpoint_dir, 'y_train.npy'), y_train)
np.save(os.path.join(args.checkpoint_dir, 'y_val.npy'), y_val)
np.save(os.path.join(args.checkpoint_dir, 'y_test.npy'), y_test)
np.save(os.path.join(args.checkpoint_dir, 'train_inds.npy'), train_inds)
np.save(os.path.join(args.checkpoint_dir, 'val_inds.npy'), val_inds)

# Model
logger.info('==> Building model..')
net_cls = get_model(args.net, args.dataset)
if 'resnet' in args.net:
    conv1 = get_conv1_params(args.dataset)
    strides = get_strides(args.dataset)
    net = net_cls(num_classes=num_classes, activation=args.activation, conv1=conv1, strides=strides)
elif args.net == 'alexnet':
    net = net_cls(num_classes=num_classes, activation=args.activation)
elif args.net == 'densenet':
    assert args.activation == 'relu'
    conv1 = get_densenet_conv1_params(args.dataset)
    net = net_cls(growth_rate=6, num_layers=52, num_classes=num_classes, drop_rate=0.0, conv1=conv1)
else:
    raise AssertionError('Does not support non Resnet architectures')
net = net.to(device)
if args.resume:
    global_state = torch.load(args.resume, map_location=torch.device(device))
    if 'best_net' in global_state:
        global_state = global_state['best_net']
    net.load_state_dict(global_state, strict=False)
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
summary(net, (img_shape[2], img_shape[0], img_shape[1]))

decay, no_decay = get_parameter_groups(net)
optimizer = optim.SGD([{'params': decay.values(), 'weight_decay': args.wd}, {'params': no_decay.values(), 'weight_decay': 0.0}],
                      lr=args.lr, momentum=args.mom, nesterov=args.mom > 0)
if args.lr_scheduler == 'multi_step':
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[50, 100, 200],
        gamma=0.1,
        verbose=True
    )
elif args.lr_scheduler == 'reduce_on_plateau':
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=metric_mode,
        factor=args.factor,
        patience=args.patience,
        verbose=True,
        cooldown=args.cooldown
    )
else:
    raise AssertionError('illegal LR scheduler {}'.format(args.lr_scheduler))

ce_criterion = nn.CrossEntropyLoss()

def loss_func(inputs, targets, kwargs=None) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    losses = {}
    outputs = net(inputs)
    losses['cross_entropy'] = ce_criterion(outputs['logits'], targets)
    losses['loss'] = losses['cross_entropy']
    return outputs, losses

def pred_func(outputs: Dict[str, torch.Tensor]) -> np.ndarray:
    _, preds = outputs['logits'].max(1)
    preds = preds.cpu().numpy()
    return preds


def train():
    """Train and validate"""
    # Training
    global global_step
    global net

    net.train()
    train_loss = 0
    predicted = []
    labels = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):  # train a single step
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, loss_dict = loss_func(inputs, targets)
        # print(loss_dict)

        loss = loss_dict['loss']
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        preds = pred_func(outputs)
        targets_np = targets.cpu().numpy()
        predicted.extend(preds)
        labels.extend(targets_np)
        num_corrected = np.sum(preds == targets_np)
        acc = num_corrected / targets.size(0)

        if global_step % 10 == 0:  # sampling, once ever 10 train iterations
            for k, v in loss_dict.items():
                train_writer.add_scalar('losses/' + k, v.item(), global_step)
            train_writer.add_scalar('metrics/acc', 100.0 * acc, global_step)
            train_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

        global_step += 1

    N = batch_idx + 1
    train_loss = train_loss / N
    predicted = np.asarray(predicted)
    labels = np.asarray(labels)
    train_acc = 100.0 * np.mean(predicted == labels)
    logger.info('Epoch #{} (TRAIN): loss={}\tacc={:.2f}'.format(epoch + 1, train_loss, train_acc))

def test():
    global net
    global best_metric
    global net

    net.eval()
    test_loss = 0
    predicted = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, loss_dict = loss_func(inputs, targets)
            preds = pred_func(outputs)

            loss = loss_dict['loss']
            test_loss += loss.item()
            predicted.extend(preds)

    N = batch_idx + 1
    test_loss = test_loss / N
    predicted = np.asarray(predicted)
    test_acc = 100.0 * np.mean(predicted == y_test)

    test_writer.add_scalar('losses/loss', test_loss, global_step)
    test_writer.add_scalar('metrics/acc', test_acc, global_step)

    if args.metric == 'accuracy':
        metric = test_acc
    elif args.metric == 'loss':
        metric = test_loss
    else:
        raise AssertionError('Unknown metric for optimization {}'.format(args.metric))

    if (args.metric == 'accuracy' and metric > best_metric) or (args.metric == 'loss' and metric < best_metric):
        best_metric = metric
        logger.info('Found new best model. Saving...')
        save_global_state()
    logger.info('Epoch #{} (TEST): loss={}\tacc={:.2f}'.format(epoch + 1, test_loss, test_acc))

    # updating learning rate if we see no improvement
    if args.lr_scheduler == 'reduce_on_plateau':
        lr_scheduler.step(metrics=metric)
    else:
        lr_scheduler.step()

def save_global_state():
    global global_state
    global_state['best_net'] = copy.deepcopy(net).state_dict()
    global_state['best_metric'] = best_metric
    global_state['epoch'] = epoch
    global_state['global_step'] = global_step
    torch.save(global_state, CHECKPOINT_PATH)

def save_current_state():
    torch.save(net.state_dict(), os.path.join(args.checkpoint_dir, 'ckpt_epoch_{}.pth'.format(epoch)))

def flush():
    train_writer.flush()
    val_writer.flush()
    test_writer.flush()
    logger.handlers[0].flush()

def load_best_net():
    global net
    global_state = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))
    net.load_state_dict(global_state['best_net'])

if __name__ == "__main__":
    best_metric = WORST_METRIC
    epoch = -1
    global_step = 0
    global_state = {}

    logger.info('Testing epoch #{}'.format(epoch + 1))
    test()

    logger.info('Start training from epoch #{} for {} epochs'.format(epoch + 1, args.epochs))
    for epoch in tqdm(range(epoch, epoch + args.epochs)):
        train()
        test()
        if epoch % 100 == 0 and epoch > 0:  # increase value for large models
            save_current_state()  # once every 10 epochs, save network to a new, distinctive checkpoint file
    save_current_state()

    # getting best metric, loading best net
    load_best_net()
    test()
    flush()
