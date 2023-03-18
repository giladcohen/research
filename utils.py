'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
import pickle
from typing import Tuple
import logging
from functools import wraps
import matplotlib.pyplot as plt
from numba import njit, jit
from typing import Dict, List, Tuple
import logging
from collections import OrderedDict
import scipy
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support

try:
    import mmcv
except:
    pass

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def remove_substr_from_keys(d: dict, substr):
    return {x.replace(substr, ''): v for x, v in d.items()}

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

# def get_np_data(data_loader: data.DataLoader) -> np.ndarray:
#     batch_size = data_loader.batch_size
#     for batch_idx, (inputs, targets) in enumerate(data_loader):
#         b = batch_idx * batch_size
#         e = b + targets.shape[0]
#         X[b:e] = inputs.cpu().numpy()

def pytorch_evaluate(net: nn.Module, x: np.ndarray, fetch_keys: List, batch_size: int, x_shape: Tuple = None,
                        output_shapes: Dict = None, to_tensor: bool=False) -> Tuple:

    if output_shapes is not None:
        for key in fetch_keys:
            assert key in output_shapes

    fetches_dict = {}
    fetches = []
    for key in fetch_keys:
        fetches_dict[key] = []

    net.eval()
    num_batch = int(np.ceil(x.shape[0]) / batch_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for m in range(num_batch):
        # Batch indexes
        begin, end = (m * batch_size, min((m + 1) * batch_size, x.shape[0]))
        input = x[begin:end]
        if x_shape is not None:
            input = input.reshape(x_shape)
        with torch.no_grad():
            outputs_dict = net(torch.from_numpy(input).to(device))
        for key in fetch_keys:
            fetches_dict[key].append(outputs_dict[key].data.cpu().detach().numpy())

    # stack variables together
    for key in fetch_keys:
        fetch = np.vstack(fetches_dict[key])
        if output_shapes is not None:
            fetch = fetch.reshape(output_shapes[key])
        if to_tensor:
            fetch = torch.as_tensor(fetch, device=torch.device(device))
        fetches.append(fetch)

    assert fetches[0].shape[0] == x.shape[0]
    return tuple(fetches)


def validate_new_inds(selected_inds: list, inds_dict: dict):
    """Validate that selected_inds are not in the train/val_inds.
       On the other hand, validate that selected_inds are all in the unlabeled_inds
    """
    new_set       = set(selected_inds)
    train_set     = set(inds_dict['train_inds'])
    val_set       = set(inds_dict['val_inds'])
    unlabeled_set = set(inds_dict['unlabeled_inds'])

    assert len(new_set.intersection(train_set)) == 0, 'Some selected samples are already in the train set'
    assert len(new_set.intersection(val_set)) == 0, 'Some selected samples are already in the val set'
    assert new_set.issubset(unlabeled_set), 'All new selected indices must be in unlabeled_inds'

def convert_norm_str_to_p(norm: str):
    assert norm in ['L1', 'L2', 'L_inf']
    if norm in ['L1', 'L2']:
        p = int(norm[-1])
    else:
        p = np.inf
    return p

def calculate_dist_mat(embeddings: np.ndarray, norm: int) -> np.ndarray:
    """Returning a distance matrix from embeddings vector"""
    kwargs = {'p': norm}
    condensed_dist = pdist(embeddings, metric='minkowski', **kwargs)
    dist_mat = squareform(condensed_dist)
    return dist_mat

def calculate_dist_mat_2(A: np.ndarray, B: np.array, norm: int) -> np.ndarray:
    """Returning a distance matrix from embeddings vector"""
    kwargs = {'p': norm}
    dist_mat = cdist(A, B, metric='minkowski', **kwargs)
    return dist_mat

def boolean_string(s):
    # to use --use_bn True or --use_bn False in the shell. See:
    # https://stackoverflow.com/questions/44561722/why-in-argparse-a-true-is-always-true
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


# network funcs
def to_1d(x):
    return x.view(x.size(0), -1)

def activation_ratio(x):
    """
    :param x: feature map. tensor of size: [batch, feature_map_size, num_pix, num_pix], where num_pix=32/16/8/4
    :return: activation ratio per 2D conv kernel. size to return value: [batch, feature_map_size]
    """
    batch_size = x.size(0)
    is_1d = len(x.size()) == 2
    if is_1d:
        spatial_size = 1
        dim = 0
    else:
        spatial_size = x.size(2) * x.size(3)
        dim = (0, 2, 3)
    activated_sum = x.sign().sum(dim=dim)
    return activated_sum / (batch_size * spatial_size)

def activation_ratio_avg(x):
    """
    :param x: feature map. tensor of size: [batch, feature_map_size, num_pix, num_pix], where num_pix=32/16/8/4
    :return: average activation ratio on all 2D conv kernels. size to return value: [batch, feature_map_size]
    """
    return x.sign().mean()

def activation_L1_ratio(x):
    """
    :param x: feature map. tensor of size: [batch, feature_map_size, num_pix, num_pix], where num_pix=32/16/8/4
    :return: average L1 activation ratio on all 2D conv kernels. size to return value: [batch, feature_map_size]
    """
    batch_size = x.size(0)
    is_1d = len(x.size()) == 2
    if is_1d:
        spatial_size = 1
        dim = 0
    else:
        spatial_size = x.size(2) * x.size(3)
        dim = (0, 2, 3)

    activated_sum = (x[x > 0]).sqrt().sum()
    activated_sum = activated_sum / (batch_size * spatial_size)
    return torch.exp(-1.0 * activated_sum)

def activation_batch_ratio(x):
    """
    :param x: feature map. tensor of size: [batch, feature_map_size, num_pix, num_pix], where num_pix=32/16/8/4
    :return: activation ratio averaged on the batch, for every pixel. size to return value: scalar
    """
    batch_size = x.size(0)
    activated_sum = x.sign().sum()
    return activated_sum / batch_size

def convert_tensor_to_image(x: np.ndarray):
    """
    :param X: np.array of size (Batch, feature_dims, H, W) or (feature_dims, H, W)
    :return: X with (Batch, H, W, feature_dims) or (H, W, feature_dims) between 0:255, uint8
    """
    X = x.copy()
    X *= 255.0
    X = np.round(X)
    X = X.astype(np.uint8)
    if len(x.shape) == 3:
        X = np.transpose(X, [1, 2, 0])
    else:
        X = np.transpose(X, [0, 2, 3, 1])
    return X

def convert_image_to_tensor(x: np.ndarray):
    """
    :param X: np.array of size (Batch, H, W, feature_dims) between 0:255, uint8
    :return: X with (Batch, feature_dims, H, W) float between [0:1]
    """
    assert x.dtype == np.uint8
    X = x.copy()
    X = X.astype(np.float32)
    X /= 255.0
    X = np.transpose(X, [0, 3, 1, 2])
    return X

def majority_vote(x):
    return np.bincount(x).argmax()

def calc_prob(x: np.ndarray) -> np.ndarray:
    return scipy.special.softmax(x)

def calc_prob_wo_l(x: np.ndarray, l: int) -> np.ndarray:
    xx = x.copy()
    xx[l] = -np.inf
    return scipy.special.softmax(xx)

def get_is_adv_prob(preds: np.ndarray) -> np.ndarray:
    # first get all the probs:
    test_size, num_points, num_classes = preds.shape
    pi_mat = scipy.special.softmax(preds, axis=2)
    pil_mat = np.zeros((preds.shape) + (num_classes,)) # (test_size(k), num_points(n), #classes(l), #classes(i))
    for cls in range(num_classes):
        tmp_preds = preds.copy()
        tmp_preds[:, :, cls] = -np.inf
        pil_mat[:, :, cls] = scipy.special.softmax(tmp_preds, axis=2)

    p_is_adv = np.nan * np.ones(preds.shape)  #shape = (k, n, i) = (test_size, num_points, num_classes)
    for k in tqdm(range(preds.shape[0])):
        l = preds[k, 0].argmax()
        for n in range(num_points):
            for i in range(num_classes):
                if i != l:
                    pi = pi_mat[k, n, i]
                    pil = pil_mat[k, n, l, i]
                    p_is_adv[k, n, i] = (1 - pi) / (1 - pi + pil)

    return p_is_adv

def get_is_adv_prob_v2(preds):
    test_size, num_points, num_classes = preds.shape
    probs = scipy.special.softmax(preds, axis=2)
    probs_mean = probs.mean(axis=1)

    probs_wo_l = np.zeros((test_size, num_points, num_classes))
    for k in range(test_size):
        l = preds[k, 0].argmax()
        for n in range(num_points):
            probs_wo_l[k, n] = calc_prob_wo_l(preds[k, n], l)
    probs_wo_l_mean = probs_wo_l.mean(axis=1)

    p_is_adv = np.zeros(test_size)
    for k in range(test_size):
        l = preds[k, 0].argmax()
        p = 0.0
        for i in range(num_classes):
            if i != l:
                p += (1 - probs_mean[k, i]) / (1 - probs_mean[k, i] + probs_wo_l_mean[k, i])
        p_is_adv[k] = p / (num_classes - 1)

    return p_is_adv

def compute_roc(y_true, y_pred, plot=False):
    """
    TODO
    :param y_true: ground truth
    :param y_pred: predictions
    :param plot:
    :return:
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score

def get_ensemble_paths(ensemble_dir):
    ensemble_subdirs = next(os.walk(ensemble_dir))[1]
    ensemble_subdirs.sort()
    ensemble_paths = []
    for j, dir in enumerate(ensemble_subdirs):  # for network j
        ensemble_paths.append(os.path.join(ensemble_dir, dir, 'ckpt.pth'))

    return ensemble_paths

def jacobian(y, x, create_graph=False):
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
    return torch.stack(jac).reshape(y.shape + x.shape)

def hessian(y, x):
    return jacobian(jacobian(y, x, create_graph=True), x)

def all_grads(y, x, create_graph=False):
    jac = torch.zeros_like(x)
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        jac[i] = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)[0][i]
        grad_y[i] = 0.
    return jac

def add_feature(x, x1):
    """Adding feature x1 to x"""
    if x is None:
        x = x1
    else:
        x = np.concatenate((x, x1), axis=1)
    return x

def save_features(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=4)


class EMAOld(object):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

class EMA(object):
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self.params = self.shadow.keys()

    def __call__(self, model):
        if self.decay > 0:
            for name, param in model.named_parameters():
                if name in self.params and param.requires_grad:
                    self.shadow[name] -= (1 - self.decay) * (self.shadow[name] - param.data)
                    param.data = self.shadow[name]

def set_logger(log_file):
    logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(log_file, mode='w'),
                                  logging.StreamHandler(sys.stdout)]
                        )

def reset_net(net):
    """This is not a robust solution and wont work for anything except core torch.nn layers:
    https://discuss.pytorch.org/t/reinitializing-the-weights-after-each-cross-validation-fold/11034/2
    """
    for layer in net.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def print_Linf_dists(X, X_test):
    logger = logging.getLogger()
    X_diff = (X - X_test).reshape(X.shape[0], -1)
    X_diff_abs = np.abs(X_diff)
    Linf_dist = X_diff_abs.max(axis=1)
    Linf_dist = Linf_dist[np.where(Linf_dist > 0.0)[0]]
    logger.info('The adversarial attacks distance: Max[L_inf]={}, E[L_inf]={}'.format(np.max(Linf_dist), np.mean(Linf_dist)))

def calc_attack_rate(y_preds: np.ndarray, y_orig_norm_preds: np.ndarray, y_gt: np.ndarray) -> float:
    """
    Args:
        y_preds: The adv image's final prediction after the defense method
        y_orig_norm_preds: The original image's predictions
        y_gt: The GT labels
        targeted: Whether or not the attack was targeted

    Returns: attack rate in %
    """
    f0_inds = []  # net_fail
    f1_inds = []  # net_succ
    f2_inds = []  # net_succ AND attack_flip

    for i in range(len(y_gt)):
        f1 = y_orig_norm_preds[i] == y_gt[i]
        f2 = f1 and y_preds[i] != y_orig_norm_preds[i]
        if f1:
            f1_inds.append(i)
        else:
            f0_inds.append(i)
        if f2:
            f2_inds.append(i)

    attack_rate = len(f2_inds) / len(f1_inds)
    return attack_rate

def get_all_files_recursive(path, suffix=None):
    files = []
    # r=root, d=directories, f=files
    for r, d, f in os.walk(path):
        for file in f:
            if suffix is None:
                files.append(os.path.join(r, file))
            elif '.' + suffix in file:
                files.append(os.path.join(r, file))
    return files

def convert_grayscale_to_rgb(x: np.ndarray) -> np.ndarray:
    """
    Converts a 2D image shape=(x, y) to a RGB image (x, y, 3).
    Args:
        x: gray image
    Returns: rgb image
    """
    return np.stack((x, ) * 3, axis=-1)

def inverse_map(x: dict) -> dict:
    """
    :param x: dictionary
    :return: inverse mapping, showing for each val its key
    """
    inv_map = OrderedDict()
    for k, v in x.items():
        inv_map[v] = k
    return inv_map

def get_image_shape(dataset: str) -> Tuple[int, int, int]:
    if dataset in ['cifar10', 'cifar100', 'svhn']:
        return 32, 32, 3
    elif dataset == 'tiny_imagenet':
        return 64, 64, 3
    else:
        raise AssertionError('Unsupported dataset {}'.format(dataset))

def get_num_classes(dataset: str) -> int:
    if dataset in ['cifar10', 'svhn']:
        return 10
    elif dataset == 'cifar100':
        return 100
    elif dataset == 'tiny_imagenet':
        return 200
    else:
        raise AssertionError('Unsupported dataset {}'.format(dataset))

def get_max_train_size(dataset: str) -> int:
    if dataset in ['cifar10', 'cifar100']:
        return 50000
    elif dataset == 'svhn':
        return 72000
    elif dataset == 'tiny_imagenet':
        return 100000
    else:
        raise AssertionError('Unsupported dataset {}'.format(dataset))

def load_characteristics(characteristics_file):
    X, Y = None, None
    data = np.load(characteristics_file)
    if X is None:
        X = data[:, :-1]
    if Y is None:
        Y = data[:, -1]  # labels only need to load once

    return X, Y

def generate_farthest_vecs(pts: np.ndarray, n: int) -> np.ndarray:
    """A Naive O(N^2) algorithm to do furthest points sampling with PyTorch"""

    assert len(pts.shape) == 2, 'pts should be 2d array'
    assert n <= pts.shape[0], 'n ({}) cannot exceed the number of input points ({})'.format(n, pts.shape[0])
    indices = np.zeros(n, dtype=np.int32)
    indices[0] = np.random.randint(pts.shape[0])
    num_selected = 1
    pts = torch.from_numpy(pts).cuda()
    distance_mat = torch.cdist(pts, pts)
    distance_mat.fill_diagonal_(0.0)

    while num_selected < n:
        all_selected_distances = distance_mat[indices[:num_selected], :]
        all_mins = all_selected_distances.min(0)[0]
        max_distance, best_index = torch.max(all_mins, 0)
        indices[num_selected] = best_index
        num_selected += 1

    return indices

def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    """Convert tensor to 3-channel images.

    Args:
        tensor (torch.Tensor): Tensor that contains multiple images, shape (
            N, C, H, W).
        mean (tuple[float], optional): Mean of images. Defaults to (0, 0, 0).
        std (tuple[float], optional): Standard deviation of images.
            Defaults to (1, 1, 1).
        to_rgb (bool, optional): Whether the tensor was converted to RGB
            format in the first place. If so, convert it back to BGR.
            Defaults to True.

    Returns:
        list[np.ndarray]: A list that contains multiple images.
    """

    if torch is None:
        raise RuntimeError('pytorch is not installed')
    assert torch.is_tensor(tensor) and tensor.ndim == 4
    assert len(mean) == 3
    assert len(std) == 3

    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = mmcv.imdenormalize(img, mean, std, to_bgr=to_rgb)
        img = np.clip(img, 0.0, 255.0)
        img = img.astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs

def get_parameter_groups(net: nn.Module) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    no_decay = dict()
    decay = dict()
    for name, m in net.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            decay[name + '.weight'] = m.weight
            decay[name + '.bias'] = m.bias
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
            no_decay[name + '.weight'] = m.weight
            no_decay[name + '.bias'] = m.bias
        else:
            if hasattr(m, 'weight'):
                no_decay[name + '.weight'] = m.weight
            if hasattr(m, 'bias'):
                no_decay[name + '.bias'] = m.bias

    # remove all None values:
    del_items = []
    for d, v in decay.items():
        if v is None:
            del_items.append(d)
    for d in del_items:
        decay.pop(d)

    del_items = []
    for d, v in no_decay.items():
        if v is None:
            del_items.append(d)
    for d in del_items:
        no_decay.pop(d)

    return decay, no_decay

def force_lr(optimizer, lr):
    """ Force a specific learning rate to all of the optimizer's weights"""
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr

def calc_acc_precision_recall(inferred_non_member, inferred_member):
    logger = logging.getLogger()
    member_acc = np.mean(inferred_member == 1)
    non_member_acc = np.mean(inferred_non_member == 0)
    acc = (member_acc * len(inferred_member) + non_member_acc * len(inferred_non_member)) / (len(inferred_member) + len(inferred_non_member))
    precision, recall, f_score, true_sum = precision_recall_fscore_support(
        y_true=np.concatenate((np.zeros(len(inferred_non_member)), np.ones(len(inferred_member)))),
        y_pred=np.concatenate((inferred_non_member, inferred_member)),
    )
    logger.info('member acc: {}, non-member acc: {}, balanced acc: {}, precision/recall(member): {}/{}, precision/recall(non-member): {}/{}'
                .format(member_acc, non_member_acc, acc, precision[1], recall[1], precision[0], recall[0]))

def load_state_dict(model: nn.Module, path: str, device='cpu') -> int:
    global_state = torch.load(path, map_location=torch.device(device))
    if 'best_net' in global_state:
        global_state = global_state['best_net']
    model.load_state_dict(global_state)
    model.to(device)
    return 1

def save_to_path(path: str, x: np.ndarray, overwrite=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path) or overwrite:
        np.save(path, x)

def normalize(x, rgb_mean, rgb_std):
    """
    :param x: np.ndaaray of image RGB of (3, W, H), normalized between [0,1]
    :param rgb_mean: Tuple of (RED mean, GREEN mean, BLUE mean)
    :param rgb_std: Tuple of (RED std, GREEN std, BLUE std)
    :return np.ndarray transformed by x = (x-mean)/std
    """
    transform = transforms.Normalize(rgb_mean, rgb_std)
    x_tensor = torch.tensor(x)
    x_new = transform(x_tensor)
    x_new = x_new.cpu().numpy()
    return x_new
