'''Test robustness with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from sklearn.neighbors import NearestNeighbors
from torchsummary import summary
import matplotlib.pyplot as plt
import scipy
import numpy as np
import json
import os
import argparse
import time
import pickle
import logging
import sys

# sys.path.insert(0, ".")
# sys.path.insert(0, "./research")
# sys.path.insert(0, "./adversarial_robustness_toolbox")

from research.datasets.train_val_test_data_loaders import get_test_loader, get_train_valid_loader, \
    get_loader_with_specific_inds, get_normalized_tensor
from research.datasets.utils import get_mini_dataset_inds, get_ensemble_dir, get_dump_dir
from research.utils import boolean_string, pytorch_evaluate, set_logger, get_ensemble_paths, \
    majority_vote, convert_tensor_to_image, print_Linf_dists, calc_attack_rate, get_image_shape
from research.models.utils import get_strides, get_conv1_params, get_model

parser = argparse.ArgumentParser(description='Evaluating robustness score')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/glove_emb/cifar100/resnet34_dim_200', type=str, help='checkpoint dir')
parser.add_argument('--checkpoint_file', default='ckpt.pth', type=str, help='checkpoint path file name')
parser.add_argument('--method', default='knn', type=str, help='softmax, knn')
parser.add_argument('--attack_dir', default='', type=str, help='attack directory, or None for normal images')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')

# for knn
parser.add_argument('--norm', default=2, help='1/2/inf')

# dump
parser.add_argument('--dump_dir', default='knn_p2', type=str, help='dump dir for logs and data')
parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)
is_attacked = args.attack_dir != ''

CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, args.checkpoint_file)
batch_size = args.batch_size

DUMP_DIR = get_dump_dir(args.checkpoint_dir, args.dump_dir, args.attack_dir)
os.makedirs(DUMP_DIR, exist_ok=True)
log_file = os.path.join(DUMP_DIR, 'log.log')

# dumping args to txt file
with open(os.path.join(DUMP_DIR, 'eval_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

set_logger(log_file)
logger = logging.getLogger()
rand_gen = np.random.RandomState(seed=12345)

dataset = train_args['dataset']
val_inds, test_inds = get_mini_dataset_inds(dataset)
val_size = len(val_inds)
test_size = len(test_inds)

# get data:
test_loader = get_test_loader(
    dataset=dataset,
    batch_size=batch_size,
    num_workers=1,
    pin_memory=device=='cuda')
img_shape = get_image_shape(dataset)
X_test = get_normalized_tensor(test_loader, img_shape, batch_size)
if not is_attacked:
    X = X_test
else:
    ATTACK_DIR = os.path.join(args.checkpoint_dir, args.attack_dir)
    X = np.load(os.path.join(ATTACK_DIR, 'X_test_adv.npy'))
    with open(os.path.join(ATTACK_DIR, 'attack_args.txt'), 'r') as f:
        attack_args = json.load(f)
    targeted = attack_args['targeted']
    y_adv = np.load(os.path.join(ATTACK_DIR, 'y_test_adv.npy')) if attack_args['targeted'] else None
    print_Linf_dists(X[test_inds], X_test[test_inds])

y_test = np.asarray(test_loader.dataset.targets)
classes = test_loader.dataset.classes
glove_vecs = test_loader.dataset.idx_to_glove_vec

# Model
logger.info('==> Building model..')
conv1 = get_conv1_params(dataset)
strides = get_strides(dataset)
global_state = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))
if 'best_net' in global_state:
    global_state = global_state['best_net']
net = get_model(train_args['net'])(num_classes=len(classes), activation=train_args['activation'], conv1=conv1,
                                   strides=strides, ext_linear=train_args.get('glove_dim', None))
net = net.to(device)
net.load_state_dict(global_state)
net.eval()  # frozen
# summary(net, (img_shape[2], img_shape[0], img_shape[1]))
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

y_gt = y_test[test_inds]

if args.method == 'softmax':
    y_preds = pytorch_evaluate(net, X, ['probs'], batch_size)[0].argmax(axis=1)[test_inds]

elif args.method == 'knn':
    if args.norm == 'inf':
        args.norm = np.inf
    knn = NearestNeighbors(n_neighbors=1, algorithm='brute', p=args.norm)
    knn.fit(glove_vecs)
    glove_embs = pytorch_evaluate(net, X, ['glove_embeddings'], batch_size)[0][test_inds]
    y_preds = knn.kneighbors(glove_embs, return_distance=False).squeeze()

acc = np.mean(y_preds == y_gt)
logger.info('Test accuracy: {}%'.format(100 * acc))
logger.handlers[0].flush()

exit(0)

# debug:
# clipping
x_clipped = torch.clip(x, 0.0, 1.0)
x_img = convert_tensor_to_image(x_clipped.detach().cpu().numpy())
for i in range(0, 5):
    plt.imshow(x_img[i])
    plt.show()
