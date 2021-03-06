'''Test robustness with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from torchsummary import summary
import matplotlib.pyplot as plt
import scipy
import numpy as np
# np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
import json
import os
import argparse
import time
import pickle
import logging
import sys
from robustbench.model_zoo.architectures.dm_wide_resnet import Swish, CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN,\
    CIFAR100_STD

# sys.path.insert(0, ".")
# sys.path.insert(0, "./research")
# sys.path.insert(0, "./adversarial_robustness_toolbox")

from research.datasets.train_val_test_data_loaders import get_test_loader, get_train_valid_loader, \
    get_loader_with_specific_inds, get_normalized_tensor
from research.datasets.utils import get_robustness_inds, get_ensemble_dir, get_dump_dir
from research.utils import boolean_string, pytorch_evaluate, set_logger, get_ensemble_paths, \
    majority_vote, convert_tensor_to_image, print_Linf_dists, calc_attack_rate, get_image_shape
from research.models.utils import get_strides, get_conv1_params, get_model

parser = argparse.ArgumentParser(description='Evaluating robustness score')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/glove_emb/cifar10/dm_wide_resnet_28_10', type=str, help='checkpoint dir')
parser.add_argument('--checkpoint_file', default='ckpt.pth', type=str, help='checkpoint path file name')
parser.add_argument('--attack_dir', default='', type=str, help='attack directory, or None for normal images')
parser.add_argument('--method', default='softmax', type=str, help='softmax/knn/cosine')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')

# for knn method
parser.add_argument('--knn_norm', default="2", type=str, help='Norm for knn: 1/2/inf')

# dump
parser.add_argument('--dump_dir', default='softmax', type=str, help='dump dir for logs and data')
parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)

if args.method != 'softmax':
    assert (train_args['glove_dim'] is not None) and (train_args['glove_dim'] != -1), 'glove_dim must be > 0'

is_attacked = args.attack_dir != ''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, args.checkpoint_file)
CLASS_EMB_VECS = os.path.join(args.checkpoint_dir, 'class_emb_vecs.npy')
batch_size = args.batch_size

DUMP_DIR = get_dump_dir(args.checkpoint_dir, args.dump_dir, args.attack_dir)
os.makedirs(DUMP_DIR, exist_ok=True)
log_file = os.path.join(DUMP_DIR, 'log.log')

# dumping args to txt file
with open(os.path.join(DUMP_DIR, 'eval_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

if args.knn_norm in ['1', '2']:
    knn_norm = int(args.knn_norm)
elif args.knn_norm == 'inf':
    knn_norm = np.inf
else:
    raise AssertionError('Unsupported norm {}'.format(args.knn_norm))

set_logger(log_file)
logger = logging.getLogger()
rand_gen = np.random.RandomState(seed=12345)

dataset = train_args['dataset']
_, test_inds = get_robustness_inds(dataset)
test_size = len(test_inds)
emb_dim = train_args.get('glove_dim', -1)
dataset_args = {'cls_to_omit': None, 'emb_selection': None, 'emb_dim': emb_dim if emb_dim != -1 else None}

# get data:
test_loader = get_test_loader(
    dataset=dataset,
    dataset_args=dataset_args,
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
    targeted = attack_args['attack'] != 'deepfool'
    y_adv = np.load(os.path.join(ATTACK_DIR, 'y_test_adv.npy')) if targeted else None
    print_Linf_dists(X[test_inds], X_test[test_inds])

y_test = np.asarray(test_loader.dataset.targets)
classes = test_loader.dataset.classes
num_classes = len(classes)
if args.method != 'softmax':
    logger.info('Loading embeddings vecs from {}'.format(CLASS_EMB_VECS))
    test_loader.dataset.overwrite_emb_vecs(np.load(CLASS_EMB_VECS))
    class_emb_vecs = test_loader.dataset.idx_to_class_emb_vec

# Model
logger.info('==> Building model..')
# if train_args['net'] == 'Rebuffi2021Fixing_70_16_cutmix_extra':
#     from robustbench.utils import load_model
#     net = load_model(model_name=train_args['net'], dataset=train_args['dataset'], threat_model='Linf',
#                      model_dir='/data/models')
#     net = net.to(device)
# else:
net_cls = get_model(train_args['net'])
if emb_dim != -1:
    ext_linear = emb_dim
else:
    ext_linear = None
if 'resnet' in train_args['net']:
    conv1 = get_conv1_params(dataset)
    strides = get_strides(dataset)
    net = net_cls(num_classes=num_classes, activation=train_args['activation'], conv1=conv1,
                  strides=strides, ext_linear=ext_linear)
else:
    net = net_cls(num_classes=num_classes, depth=28, width=10, activation_fn=Swish,
                  mean=CIFAR10_MEAN, std=CIFAR10_STD, ext_linear=ext_linear)
net = net.to(device)
global_state = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))
if 'best_net' in global_state:
    global_state = global_state['best_net']
net.load_state_dict(global_state)

net.eval()  # frozen
# summary(net, (img_shape[2], img_shape[0], img_shape[1]))
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

y_gt = y_test[test_inds]

if args.method == 'softmax':
    y_probs = pytorch_evaluate(net, X, ['probs'], batch_size)[0][test_inds]
    y_preds = y_probs.argmax(axis=1)

elif args.method == 'knn':
    knn = NearestNeighbors(n_neighbors=1, algorithm='brute', p=knn_norm)
    knn.fit(class_emb_vecs)
    glove_embs = pytorch_evaluate(net, X, ['glove_embeddings'], batch_size)[0][test_inds]
    y_preds = knn.kneighbors(glove_embs, return_distance=False).squeeze()

elif args.method == 'cosine':
    cos = nn.CosineSimilarity()
    glove_embs = pytorch_evaluate(net, X, ['glove_embeddings'], batch_size, to_tensor=True)[0][test_inds]
    distance_mat = torch.zeros((test_size, num_classes)).to(device)
    for cls_idx in range(num_classes):
        embs = np.tile(class_emb_vecs[cls_idx], (test_size, 1))
        embs = torch.from_numpy(embs).to(device)
        distance_mat[:, cls_idx] = cos(glove_embs, embs)
    distance_mat = distance_mat.cpu().numpy()
    y_preds = distance_mat.argmax(1)

acc = np.mean(y_gt == y_preds)
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
