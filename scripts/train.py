'''Train DNNs with GloVe via PyTorch.'''
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchsummary import summary
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
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from robustbench.model_zoo.architectures.dm_wide_resnet import Swish, CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, \
    CIFAR100_STD

from research.losses.losses import LinfLoss, L2Loss, CosineEmbeddingLossV2, TradesLoss, VATLoss, GuidedAdversarialTrainingLoss
from research.datasets.train_val_test_data_loaders import get_test_loader, get_train_valid_loader
from research.utils import boolean_string, get_image_shape, set_logger
from research.models.utils import get_strides, get_conv1_params, get_model

parser = argparse.ArgumentParser(description='Training networks using PyTorch')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset: cifar10, cifar100, svhn, tiny_imagenet')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/glove_emb/cifar10/debug', type=str, help='checkpoint dir')
parser.add_argument('--glove', default=False, type=boolean_string, help='Train using GloVe embeddings instead of CE')

# architecture:
parser.add_argument('--net', default='resnet18', type=str, help='network architecture')
parser.add_argument('--activation', default='relu', type=str, help='network activation: relu or softplus')
parser.add_argument('--glove_dim', default=1024, type=int, help='Size of the words embeddings. -1 for no layer')

# cross entropy with embedding loss
parser.add_argument('--aux_loss', default=False, type=boolean_string,
                    help='Train cross entropy with embeddings loss as auxiliary loss')
parser.add_argument('--w_aux', default=0.5, type=float, help="The embedding loss auxiliary loss's weight")

# GloVe settings
parser.add_argument('--emb_selection', default=None, type=str, help='Selection of glove embeddings: glove/random/farthest_points/orthogonal')
parser.add_argument('--emb_loss', default='cosine', type=str, help='The loss used for embedding training: L1/L2/Linf/cosine')
parser.add_argument('--eval_method', default='softmax', type=str, help='eval method for embeddings: softmax/knn/cosine')
parser.add_argument('--knn_norm', default='2', type=str, help='Norm for knn: 1/2/inf')

# optimization:
parser.add_argument('--resume', default=None, type=str, help='Path to checkpoint to be resumed')
parser.add_argument('--mom', default=0.9, type=float, help='weight momentum of SGD optimizer')
parser.add_argument('--epochs', default='300', type=int, help='number of epochs')
parser.add_argument('--wd', default=0.0001, type=float, help='weight decay')  # was 5e-4 for batch_size=128
parser.add_argument('--val_size', default=0.05, type=float, help='Fraction of validation size')
parser.add_argument('--num_workers', default=4, type=int, help='Data loading threads')
parser.add_argument('--metric', default='accuracy', type=str, help='metric to optimize. accuracy or sparsity')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_only_embs', default=False, type=boolean_string, help='Training only the ext_linear weights/bias')

# LR schedule
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--factor', default=0.9, type=float, help='LR schedule factor')
parser.add_argument('--patience', default=3, type=int, help='LR schedule patience')
parser.add_argument('--cooldown', default=0, type=int, help='LR cooldown')

# common TRADES/VAT/GAT params
parser.add_argument('--adv_trades', default=False, type=boolean_string, help='Use adv robust training using TRADES')
parser.add_argument('--adv_vat', default=False, type=boolean_string, help='Use virtual adversarial training')
parser.add_argument('--adv_gat', default=False, type=boolean_string, help='Use GAT adversarial training')
parser.add_argument('--adv_emb_loss', default=None, type=str, help='criterion for the adversarial loss in TRADES')
parser.add_argument('--epsilon', default=8, type=float, help='epsilon for TRADES loss')
parser.add_argument('--step_size', default=0.007, type=float, help='step size for TRADES loss')
# VAT params
parser.add_argument('--beta', default=1, type=float, help='weight for adversarial loss during training (alpha for VAT)')
parser.add_argument('--xi', default=10, type=float, help='xi param for VAT')
# GAT params
parser.add_argument('--bern_eps', default=4, type=float, help='Bernoulli noise for GAT adv training')
parser.add_argument('--steps', default=1, type=int, help='number of steps for GAT')
parser.add_argument('--l2_reg', default=1.0, type=float, help='L2 regularization coefficient for GAT')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

if args.epsilon > 1:
    args.epsilon = args.epsilon / 255
if args.bern_eps > 1:
    args.bern_eps = args.bern_eps / 255

assert args.adv_trades + args.adv_vat + args.adv_gat <= 1, 'TRADES/VAT/GAT cannot be set together'

if args.aux_loss:
    assert args.glove, 'using auxiliary loss requires glove=True'

if args.glove:
    assert args.glove_dim != -1, 'glove_dim must be set when training with glove embeddings'

# dumping args to txt file
os.makedirs(args.checkpoint_dir, exist_ok=True)
with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

if args.knn_norm in ['1', '2']:
    knn_norm = int(args.knn_norm)
elif args.knn_norm == 'inf':
    knn_norm = np.inf
else:
    raise AssertionError('Unsupported norm {}'.format(args.knn_norm))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')
CLASS_EMB_VECS = os.path.join(args.checkpoint_dir, 'class_emb_vecs.npy')
log_file = os.path.join(args.checkpoint_dir, 'log.log')
batch_size = args.batch_size

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
dataset_args = {'cls_to_omit': None, 'emb_selection': args.emb_selection,
                'emb_dim': args.glove_dim if args.glove_dim != -1 else None}
trainloader, valloader, train_inds, val_inds = get_train_valid_loader(
    dataset=args.dataset,
    dataset_args=dataset_args,
    batch_size=batch_size,
    rand_gen=rand_gen,
    valid_size=args.val_size,
    num_workers=args.num_workers,
    pin_memory=device=='cuda'
)
testloader = get_test_loader(
    dataset=args.dataset,
    dataset_args=dataset_args,
    batch_size=batch_size,
    num_workers=args.num_workers,
    pin_memory=device=='cuda'
)

img_shape = get_image_shape(args.dataset)
classes = trainloader.dataset.classes
num_classes = len(classes)
class_emb_vecs = trainloader.dataset.idx_to_class_emb_vec
valloader.dataset.overwrite_emb_vecs(class_emb_vecs)
testloader.dataset.overwrite_emb_vecs(class_emb_vecs)
train_size = len(trainloader.dataset)
val_size   = len(valloader.dataset)
test_size  = len(testloader.dataset)

if args.glove:
    # saving glove_vecs for the classes:
    np.save(CLASS_EMB_VECS, class_emb_vecs)

# Model
logger.info('==> Building model..')
net_cls = get_model(args.net)
ext_linear = args.glove_dim if args.glove_dim != -1 else None
if 'resnet' in args.net:
    conv1 = get_conv1_params(args.dataset)
    strides = get_strides(args.dataset)
    net = net_cls(num_classes=num_classes, activation=args.activation, conv1=conv1, strides=strides,
                  ext_linear=ext_linear)
else:
    net = net_cls(num_classes=num_classes, depth=28, width=10, activation_fn=Swish,
                  mean=CIFAR10_MEAN, std=CIFAR10_STD, ext_linear=ext_linear)
net = net.to(device)
if args.resume:
    global_state = torch.load(args.resume, map_location=torch.device(device))
    if 'best_net' in global_state:
        global_state = global_state['best_net']
    net.load_state_dict(global_state, strict=False)

summary(net, (img_shape[2], img_shape[0], img_shape[1]))

if args.glove and args.eval_method == 'knn':
    # knn model
    knn = NearestNeighbors(n_neighbors=1, algorithm='brute', p=knn_norm)
    knn.fit(class_emb_vecs)
else:
    knn = None

if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

ce_criterion = nn.CrossEntropyLoss()
cos = nn.CosineSimilarity()
y_train    = np.asarray(trainloader.dataset.targets)
y_val      = np.asarray(valloader.dataset.targets)
y_test     = np.asarray(testloader.dataset.targets)

# dump to dir:
np.save(os.path.join(args.checkpoint_dir, 'y_train.npy'), y_train)
np.save(os.path.join(args.checkpoint_dir, 'y_val.npy'), y_val)
np.save(os.path.join(args.checkpoint_dir, 'y_test.npy'), y_test)
np.save(os.path.join(args.checkpoint_dir, 'train_inds.npy'), train_inds)
np.save(os.path.join(args.checkpoint_dir, 'val_inds.npy'), val_inds)

# setting parameter groups:
no_decay = dict()
decay = dict()
for name, m in net.named_modules():
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        decay[name + '.weight'] = m.weight
        decay[name + '.bias'] = m.bias
    elif isinstance(m, nn.BatchNorm2d):
        no_decay[name + '.weight'] = m.weight
        no_decay[name + '.bias'] = m.bias
    else:
        if hasattr(m, 'weight'):
            no_decay[name + '.weight'] = m.weight
        if hasattr(m, 'bias'):
            no_decay[name + '.bias'] = m.weight

# remove all None values:
del_items = []
for d, v in decay.items():
    if v is None:
        del_items.append(d)
for d in del_items:
    print('deleting {} from decay'.format(d))
    decay.pop(d)

del_items = []
for d, v in no_decay.items():
    if v is None:
        del_items.append(d)
for d in del_items:
    print('deleting {} from no_decay'.format(d))
    no_decay.pop(d)
# print(decay.keys())
# print(no_decay.keys())

if args.train_only_embs:
    optimizer = optim.SGD(net.ext_linear.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd, nesterov=args.mom > 0)
else:
    optimizer = optim.SGD([{'params': decay.values(), 'weight_decay': args.wd}, {'params': no_decay.values(), 'weight_decay': 0.0}]
                          , lr=args.lr, momentum=args.mom, nesterov=args.mom > 0)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode=metric_mode,
    factor=args.factor,
    patience=args.patience,
    verbose=True,
    cooldown=args.cooldown
)

if args.emb_loss == 'L1':
    emb_loss = nn.L1Loss()
elif args.emb_loss == 'L2':
    emb_loss = L2Loss()
elif args.emb_loss == 'Linf':
    emb_loss = LinfLoss()
elif args.emb_loss == 'cosine':
    emb_loss = CosineEmbeddingLossV2()
else:
    raise AssertionError('Unknown value args.emb_loss = {}'.format(args.emb_loss))

if args.adv_emb_loss is None:
    adv_emb_loss = args.emb_loss
else:
    adv_emb_loss = args.adv_emb_loss

if args.adv_trades:
    trades_loss = TradesLoss(
        model=net,
        optimizer=optimizer,
        step_size=args.step_size,
        epsilon=args.epsilon,
        perturb_steps=10,
        beta=args.beta,
        field='glove_embeddings' if args.glove else 'logits',
        criterion=args.emb_loss if args.glove else 'ce',
        adv_criterion=adv_emb_loss if args.glove else 'kl'
    )
elif args.adv_vat:
    vat_loss = VATLoss(
        model=net,
        field='glove_embeddings' if args.glove else 'logits',
        adv_criterion=adv_emb_loss if args.glove else 'kl',
        xi=args.xi,
        eps=args.epsilon
    )
elif args.adv_gat:
    gat_loss = GuidedAdversarialTrainingLoss(
        model=net,
        eps=args.epsilon,
        bern_eps=args.bern_eps,
        steps=args.steps,
        l2_reg=args.l2_reg,
        field='glove_embeddings' if args.glove else 'logits',
        criterion=args.emb_loss if args.glove else 'ce',
        adv_criterion=adv_emb_loss if args.glove else 'ce'
    )


def targets_to_embs(targets):
    return torch.from_numpy(class_emb_vecs[targets.cpu()]).to(device)

def output_loss_trades(inputs, targets, kwargs) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    if args.glove:
        targets = targets_to_embs(targets)
    return trades_loss(inputs, targets, kwargs)

def output_loss_vat(inputs, targets, kwargs) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    losses = {}
    loss_robust, outputs = vat_loss(inputs, kwargs)
    if args.glove:
        targets = targets_to_embs(targets)
        loss_natural = emb_loss(outputs['glove_embeddings'], targets)
    else:
        loss_natural = ce_criterion(outputs['logits'], targets)
    loss = loss_natural + args.beta * loss_robust
    losses['natural'] = loss_natural
    losses['robust'] = loss_robust
    losses['loss'] = loss
    return outputs, losses

def output_loss_gat(inputs, targets, kwargs) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    if args.glove:
        targets = targets_to_embs(targets)
    return gat_loss(inputs, targets, kwargs)

def output_loss_emb(inputs, targets, kwargs) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    losses = {}
    embs = targets_to_embs(targets)
    outputs = net(inputs)
    loss = emb_loss(outputs['glove_embeddings'], embs)
    losses['embeddings'] = loss
    losses['loss'] = loss
    return outputs, losses

def output_loss_normal(inputs, targets, kwargs) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    losses = {}
    outputs = net(inputs)
    loss = ce_criterion(outputs['logits'], targets)
    losses['cross_entropy'] = loss
    losses['loss'] = loss
    return outputs, losses

def output_loss_aux(inputs, targets, kwargs) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    losses = {}
    embs = targets_to_embs(targets)
    outputs = net(inputs)
    loss_ce = ce_criterion(outputs['logits'], targets)
    loss_emb = emb_loss(outputs['glove_embeddings'], embs)
    loss = (1 - args.w_aux) * loss_ce + args.w_aux * loss_emb
    losses['cross_entropy'] = loss_ce
    losses['embeddings'] = loss_emb
    losses['loss'] = loss
    return outputs, losses

def softmax_pred(outputs: Dict[str, torch.Tensor]) -> np.ndarray:
    _, preds = outputs['logits'].max(1)
    preds = preds.cpu().numpy()
    return preds

def get_loss():
    if args.adv_trades:
        loss_func = output_loss_trades
    elif args.adv_vat:
        loss_func = output_loss_vat
    elif args.adv_gat:
        loss_func = output_loss_gat
    elif args.aux_loss:
        loss_func = output_loss_aux
    elif args.glove:
        loss_func = output_loss_emb
    else:
        loss_func = output_loss_normal
    return loss_func

def knn_pred(outputs: Dict[str, torch.Tensor]) -> np.ndarray:
    preds = knn.kneighbors(outputs['glove_embeddings'].detach().cpu().numpy(), return_distance=False).squeeze()
    return preds

def cosine_pred(outputs: Dict[str, torch.Tensor]) -> np.ndarray:
    glove_embs = outputs['glove_embeddings']
    bs = glove_embs.size(0)
    distance_mat = torch.zeros((bs, num_classes)).to(device)
    for cls_idx in range(num_classes):
        embs = np.tile(class_emb_vecs[cls_idx], (bs, 1))
        embs = torch.from_numpy(embs).to(device)
        distance_mat[:, cls_idx] = cos(glove_embs, embs)
    distance_mat = distance_mat.detach().cpu().numpy()
    preds = distance_mat.argmax(1)
    return preds

def get_pred():
    if args.eval_method == 'softmax':
        pred_func = softmax_pred
    elif args.eval_method == 'knn':
        pred_func = knn_pred
    elif args.eval_method == 'cosine':
        pred_func = cosine_pred
    else:
        raise AssertionError('Unknown args.glove=True with unknown eval_method={}'.format(args.eval_method))
    return pred_func


loss_func = get_loss()
pred_func = get_pred()

def train():
    """Train and validate"""
    # Training
    global global_step
    global epoch
    global net

    net.train()
    train_loss = 0
    predicted = []
    labels = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):  # train a single step
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        kwargs = {'is_training': True, 'alt': batch_idx % 2}
        outputs, loss_dict = loss_func(inputs, targets, kwargs)

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

        if global_step % 11 == 0:  # sampling, once ever 10 train iterations
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

def validate():
    global global_step
    global global_state
    global best_metric
    global epoch
    global net

    net.eval()
    val_loss = 0
    predicted = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            kwargs = {'is_training': False, 'alt': 0}
            outputs, loss_dict = loss_func(inputs, targets, kwargs)
            preds = pred_func(outputs)

            loss = loss_dict['loss']
            val_loss += loss.item()
            predicted.extend(preds)

    N = batch_idx + 1
    val_loss    = val_loss / N
    predicted = np.asarray(predicted)
    val_acc = 100.0 * np.mean(predicted == y_val)

    val_writer.add_scalar('losses/loss', val_loss, global_step)
    val_writer.add_scalar('metrics/acc', val_acc, global_step)

    if args.metric == 'accuracy':
        metric = val_acc
    elif args.metric == 'loss':
        metric = val_loss
    else:
        raise AssertionError('Unknown metric for optimization {}'.format(args.metric))

    if (args.metric == 'accuracy' and metric > best_metric) or (args.metric == 'loss' and metric < best_metric):
        best_metric = metric
        logger.info('Found new best model. Saving...')
        save_global_state()
    logger.info('Epoch #{} (VAL): loss={}\tacc={:.2f}\tbest_metric({})={}'.format(epoch + 1, val_loss, val_acc, args.metric, best_metric))

    # updating learning rate if we see no improvement
    lr_scheduler.step(metrics=metric)

def test():
    global global_step
    global epoch
    global net

    net.eval()
    test_loss = 0
    predicted = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            kwargs = {'is_training': False, 'alt': 0}
            outputs, loss_dict = loss_func(inputs, targets, kwargs)
            preds = pred_func(outputs)

            loss = loss_dict['loss']
            test_loss += loss.item()
            predicted.extend(preds)

    N = batch_idx + 1
    test_loss    = test_loss / N
    predicted = np.asarray(predicted)
    test_acc = 100.0 * np.mean(predicted == y_test)

    test_writer.add_scalar('losses/loss', test_loss, global_step)
    test_writer.add_scalar('metrics/acc', test_acc, global_step)

    logger.info('Epoch #{} (TEST): loss={}\tacc={:.2f}'.format(epoch + 1, test_loss, test_acc))

def save_global_state():
    global global_state, net, best_metric, epoch, global_step
    global_state['best_net'] = copy.deepcopy(net).state_dict()
    global_state['best_metric'] = best_metric
    global_state['epoch'] = epoch
    global_state['global_step'] = global_step
    torch.save(global_state, CHECKPOINT_PATH)

def save_current_state():
    global epoch
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
    epoch = 0
    global_step = 0
    global_state = {}

    logger.info('Testing epoch #{}'.format(epoch + 1))
    test()

    logger.info('Start training from epoch #{} for {} epochs'.format(epoch + 1, args.epochs))
    for epoch in tqdm(range(epoch, epoch + args.epochs)):
        train()
        validate()
        if epoch % 10 == 0 and epoch > 0:
            test()
            if epoch % 10 == 0:
                save_current_state()  # once every 10 epochs, save network to a new, distinctive checkpoint file
    save_current_state()

    # getting best metric, loading best net
    load_best_net()
    test()
    flush()
