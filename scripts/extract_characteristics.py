'''Extract fetures for adversarial detection.'''
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
from scipy.optimize import minimize
import sys

# sys.path.insert(0, ".")
# sys.path.insert(0, "./research")
# sys.path.insert(0, "./adversarial_robustness_toolbox")
from research.datasets.train_val_test_data_loaders import get_test_loader, get_train_valid_loader, \
    get_loader_with_specific_inds, get_normalized_tensor
from research.datasets.utils import get_detection_inds, get_ensemble_dir, get_dump_dir
from research.utils import boolean_string, pytorch_evaluate, set_logger, get_ensemble_paths, \
    majority_vote, convert_tensor_to_image, print_Linf_dists, calc_attack_rate, get_image_shape
from research.models.utils import get_strides, get_conv1_params, get_model

parser = argparse.ArgumentParser(description='Evaluating robustness score')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/glove_emb/cifar10/resnet34_glove_p2', type=str, help='checkpoint dir')
parser.add_argument('--checkpoint_file', default='ckpt.pth', type=str, help='checkpoint path file name')
parser.add_argument('--attack_dir', default='fgsm_L2', type=str, help='attack directory')
parser.add_argument('--eval_method', default='knn', type=str, help='softmax/knn/cosine')
parser.add_argument('--detect_method', default='mahalanobis', type=str, help='lid/mahalanobis/dknn')
parser.add_argument('--dump_dir', default='debug', type=str, help='dump dir for logs and characteristics')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')

# for knn norm
parser.add_argument('--norm', default="2", type=str, help='Norm for knn: 1/2/inf')

# for lid/dknn
parser.add_argument('--k_nearest', default=-1, help='number of nearest neighbors to use for LID/dknn detection')

# for mahalanobis
parser.add_argument('--magnitude', default=-1, help='magnitude for mahalanobis detection')
parser.add_argument('--rgb_scale', default=1, help='scale for mahalanobis')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, args.checkpoint_file)
ATTACK_DIR = os.path.join(args.checkpoint_dir, args.attack_dir)
DUMP_DIR = os.path.join(ATTACK_DIR, args.dump_dir)
PLOTS_DIR = os.path.join(DUMP_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)
batch_size = args.batch_size
log_file = os.path.join(DUMP_DIR, 'log.log')
with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)
with open(os.path.join(ATTACK_DIR, 'attack_args.txt'), 'r') as f:
    attack_args = json.load(f)
# dumping args to txt file
with open(os.path.join(DUMP_DIR, 'eval_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

if args.norm in ['1', '2']:
    args.norm = int(args.norm)
elif args.norm == 'inf':
    args.norm = np.inf
else:
    raise AssertionError('Unsupported norm {}'.format(args.norm))

set_logger(log_file)
logger = logging.getLogger()
rand_gen = np.random.RandomState(seed=12345)
targeted = attack_args['attack'] != 'deepfool'
dataset = train_args['dataset']
val_inds, test_inds = get_detection_inds(dataset)
val_size = len(val_inds)
test_size = len(test_inds)

# get data:
test_loader = get_test_loader(
    dataset=dataset,
    batch_size=batch_size,
    num_workers=1,
    pin_memory=device=='cuda')
img_shape = get_image_shape(dataset)

X_normal     = get_normalized_tensor(test_loader, img_shape, batch_size)
X_val        = X_normal[val_inds]
X_test       = X_normal[test_inds]

y_normal     = np.asarray(test_loader.dataset.targets)
y_val        = y_normal[val_inds]
y_test       = y_normal[test_inds]

X_adv        = np.load(os.path.join(ATTACK_DIR, 'X_test_adv.npy'))
X_adv_val    = X_adv[val_inds]
X_adv_test   = X_adv[test_inds]

if targeted:
    y_adv = np.load(os.path.join(ATTACK_DIR, 'y_test_adv.npy'))
    y_adv_val  = y_adv[val_inds]
    y_adv_test = y_adv[test_inds]
else:
    y_adv = None
    y_adv_val = None
    y_adv_test = None

classes = test_loader.dataset.classes
num_classes = len(classes)
glove_vecs = test_loader.dataset.idx_to_glove_vec

# Model
logger.info('==> Building model..')
conv1 = get_conv1_params(dataset)
strides = get_strides(dataset)
glove_dim = train_args.get('glove_dim', None)
global_state = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))
if 'best_net' in global_state:
    global_state = global_state['best_net']
net = get_model(train_args['net'])(num_classes=num_classes, activation=train_args['activation'], conv1=conv1,
                                   strides=strides, ext_linear=glove_dim)
net = net.to(device)
net.load_state_dict(global_state)
net.eval()  # frozen
# summary(net, (img_shape[2], img_shape[0], img_shape[1]))
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# get noisy samples
def get_noisy_samples(x, std):
    """ Add Gaussian noise to the samples """
    X_noisy = np.clip(x + rand_gen.normal(loc=0.0, scale=std, size=x.shape), 0, 1)
    return X_noisy

def get_l2_dist(x1, x2):
    assert x1.shape == x2.shape
    diff = x1.reshape((len(x1), -1)) - x2.reshape((len(x2), -1))
    l2_dist = np.linalg.norm(diff, ord=2, axis=1).mean()
    return l2_dist

def minimize_noise_diff(std, X_test, l2_dist_adv):
    X_noisy = get_noisy_samples(X_test, std)
    l2_dist = get_l2_dist(X_noisy, X_test)
    ret = np.abs(l2_dist - l2_dist_adv)
    return ret

def generate_noisy_samples(X_test, X_adv):
    logger.info('Generating X_noisy...')
    l2_dist_adv = get_l2_dist(X_adv, X_test)
    # Now we need to generate X_noisy so that the diff between <X_noisy, X_test> would be the resulted l2_dist_adv
    res = minimize(minimize_noise_diff, np.array([0.008]), args=(X_test, l2_dist_adv), tol=1e-4, method='nelder-mead')
    std = float(res.x)
    X_noisy = get_noisy_samples(X_test, std)
    l2_dist_noisy = get_l2_dist(X_noisy, X_test)
    logger.info('Got X_noisy with L2 dist of {}. The adversarial L2 dist is {}'.format(l2_dist_noisy, l2_dist_adv))
    return std, X_noisy


if not os.path.exists(os.path.join(ATTACK_DIR, 'X_det_noisy_val.npy')):
    std, X_noisy_val = generate_noisy_samples(X_val, X_adv_val)
    X_noisy_test = get_noisy_samples(X_test, std)
    X_noisy_val, X_noisy_test = X_noisy_val.astype(np.float32), X_noisy_test.astype(np.float32)
    np.save(os.path.join(ATTACK_DIR, 'X_det_noisy_val.npy'), X_noisy_val)
    np.save(os.path.join(ATTACK_DIR, 'X_det_noisy_test.npy'), X_noisy_test)
else:
    X_noisy_val  = np.load(os.path.join(ATTACK_DIR, 'X_det_noisy_val.npy'))
    X_noisy_test = np.load(os.path.join(ATTACK_DIR, 'X_det_noisy_test.npy'))

def eval(x):
    if args.eval_method == 'softmax':
        y_probs = pytorch_evaluate(net, x, ['probs'], batch_size)[0]
        y_preds = y_probs.argmax(axis=1)
    elif args.eval_method == 'knn':
        knn = NearestNeighbors(n_neighbors=1, algorithm='brute', p=args.norm)
        knn.fit(glove_vecs)
        glove_embs = pytorch_evaluate(net, x, ['glove_embeddings'], batch_size)[0]
        y_preds = knn.kneighbors(glove_embs, return_distance=False).squeeze()
    else:
        raise AssertionError('Unrecognized value of args.eval_method: {}'.format(args.eval_method))
    return y_preds

y_preds_val = eval(X_val)
y_preds_test = eval(X_test)
logger.info('Classification accuracy over all val samples: {}%'.format(100 * np.mean(y_preds_val == y_val)))
logger.info('Classification accuracy over all test samples: {}%'.format(100 * np.mean(y_preds_test == y_test)))
val_inds_correct = np.where(y_preds_val == y_val)[0]
test_inds_correct = np.where(y_preds_test == y_test)[0]

# filtering only correct detections
# val
X_val       = X_val[val_inds_correct]
X_adv_val   = X_adv_val[val_inds_correct]
X_noisy_val = X_noisy_val[val_inds_correct]
y_val       = y_val[val_inds_correct]
y_adv_val   = y_adv_val[val_inds_correct] if targeted else None
logger.info('X_val: {}\nX_adv_val: {}\nX_noisy_val: {}'.format(X_val.shape, X_adv_val.shape, X_noisy_val.shape))
# test
X_test       = X_test[test_inds_correct]
X_adv_test   = X_adv_test[test_inds_correct]
X_noisy_test = X_noisy_test[test_inds_correct]
y_test       = y_test[test_inds_correct]
y_adv_test   = y_adv_test[test_inds_correct] if targeted else None
logger.info('X_test: {}\nX_adv_test: {}\nX_noisy_test: {}'.format(X_test.shape, X_adv_test.shape, X_noisy_test.shape))


logger.handlers[0].flush()
exit(0)

# debug:
# clipping
x_clipped = torch.clip(x, 0.0, 1.0)
x_img = convert_tensor_to_image(x_clipped.detach().cpu().numpy())
for i in range(0, 5):
    plt.imshow(x_img[i])
    plt.show()
