'''Extract fetures for adversarial detection.'''
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
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
from sklearn.covariance import EmpiricalCovariance
import sys
# sys.path.insert(0, ".")
# sys.path.insert(0, "./research")
# sys.path.insert(0, "./adversarial_robustness_toolbox")
from research.datasets.train_val_test_data_loaders import get_test_loader, get_train_valid_loader, \
    get_loader_with_specific_inds, get_normalized_tensor, get_all_data_loader
from research.datasets.utils import get_detection_inds, get_ensemble_dir, get_dump_dir
from research.utils import boolean_string, pytorch_evaluate, set_logger, get_ensemble_paths, \
    majority_vote, convert_tensor_to_image, print_Linf_dists, calc_attack_rate, get_image_shape, inverse_map
from research.models.utils import get_strides, get_conv1_params, get_model

parser = argparse.ArgumentParser(description='Evaluating robustness score')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/glove_emb/cifar10/resnet34_dim_200', type=str, help='checkpoint dir')
parser.add_argument('--checkpoint_file', default='ckpt.pth', type=str, help='checkpoint path file name')
parser.add_argument('--attack_dir', default='pgd', type=str, help='attack directory')
parser.add_argument('--eval_method', default='softmax', type=str, help='softmax/knn/cosine')
parser.add_argument('--detect_method', default='mahalanobis', type=str, help='lid/mahalanobis/dknn')
parser.add_argument('--include_noise', default=True, type=boolean_string, help='include X_noise in characteristics')
parser.add_argument('--dump_dir', default='debug', type=str, help='dump dir for logs and characteristics')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')

# for knn norm
parser.add_argument('--norm', default="2", type=str, help='Norm for knn: 1/2/inf')

# for lid/dknn
parser.add_argument('--k_nearest', default=-1, type=int, help='number of nearest neighbors to use for LID/dknn detection')

# for mahalanobis
parser.add_argument('--magnitude', default=-1, type=float, help='magnitude for mahalanobis detection')
parser.add_argument('--use_raw_grads', default=False, type=boolean_string, help='Use raw grads without taking their sign values')
parser.add_argument('--rgb_scale', default=1, type=float, help='scale for mahalanobis')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

ATTACK_DIR = os.path.join(args.checkpoint_dir, args.attack_dir)
with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)
with open(os.path.join(ATTACK_DIR, 'attack_args.txt'), 'r') as f:
    attack_args = json.load(f)

if args.eval_method == 'softmax':
    assert not train_args['glove'], 'softmax evaluation can only be done for non GloVe training'
else:  # knn/cosine
    assert train_args['glove'], 'knn/cosine evaluation can only be done for GloVe training'
    assert (train_args['glove_dim'] is not None) and (train_args['glove_dim'] != -1), 'glove_dim must be > 0'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, args.checkpoint_file)
CLASS_EMB_VECS = os.path.join(args.checkpoint_dir, 'class_emb_vecs.npy')
DUMP_DIR = os.path.join(ATTACK_DIR, args.dump_dir)
PLOTS_DIR = os.path.join(DUMP_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)
batch_size = args.batch_size
log_file = os.path.join(DUMP_DIR, 'log.log')
# dumping args to txt file
with open(os.path.join(DUMP_DIR, 'extract_characteristics_args.txt'), 'w') as f:
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
dataset_args = {'cls_to_omit': None, 'emb_selection': train_args.get('args.emb_selection', None)}

# get data:
train_loader = get_all_data_loader(
    dataset=dataset,
    dataset_args=dataset_args,
    batch_size=batch_size,
    num_workers=0,
    pin_memory=False,
)
test_loader = get_test_loader(
    dataset=dataset,
    dataset_args=dataset_args,
    batch_size=batch_size,
    num_workers=0,
    pin_memory=False)
img_shape = get_image_shape(dataset)

X_train      = get_normalized_tensor(train_loader, img_shape, batch_size)
y_train      = np.asarray(train_loader.dataset.targets)

X_normal     = get_normalized_tensor(test_loader, img_shape, batch_size)
y_normal     = np.asarray(test_loader.dataset.targets)

X_adv        = np.load(os.path.join(ATTACK_DIR, 'X_test_adv.npy'))
y_adv        = np.load(os.path.join(ATTACK_DIR, 'y_test_adv.npy')) if targeted else None

# separating to val and test:
X_val        = X_normal[val_inds]
X_test       = X_normal[test_inds]

y_val        = y_normal[val_inds]
y_test       = y_normal[test_inds]

X_adv_val    = X_adv[val_inds]
X_adv_test   = X_adv[test_inds]

y_adv_val    = y_adv[val_inds] if targeted else None
y_adv_test   = y_adv[test_inds] if targeted else None

classes = test_loader.dataset.classes
num_classes = len(classes)
train_loader.dataset.overwrite_emb_vecs(np.load(CLASS_EMB_VECS))
test_loader.dataset.overwrite_emb_vecs(np.load(CLASS_EMB_VECS))
class_emb_vecs = test_loader.dataset.idx_to_class_emb_vec

# Model
logger.info('==> Building model..')
conv1 = get_conv1_params(dataset)
strides = get_strides(dataset)
glove_dim = train_args.get('glove_dim', -1)
if glove_dim != -1:
    ext_linear = glove_dim
else:
    ext_linear = None
net = get_model(train_args['net'])(num_classes=num_classes, activation=train_args['activation'], conv1=conv1,
                                   strides=strides, ext_linear=ext_linear)
net = net.to(device)
global_state = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))
if 'best_net' in global_state:
    global_state = global_state['best_net']
net.load_state_dict(global_state)
net.eval()  # frozen
# summary(net, (img_shape[2], img_shape[0], img_shape[1]))


# if train_args['glove']: # using glove training. Use embeddings and glove_embeddings only
#     layer_to_idx = OrderedDict([('embeddings', 0), ('glove_embeddings', 1)])
# elif glove_dim != -1:  # not training with GloVe, but using extended architecture
#     layer_to_idx = OrderedDict([('embeddings', 0), ('glove_embeddings', 1), ('logits', 2)])
# else:  # not training with Glove, with standard architecture
#     layer_to_idx = OrderedDict([('embeddings', 0), ('logits', 1)])
layer_to_idx = OrderedDict([('glove_embeddings', 0)])
layer_to_size = OrderedDict([('embeddings', net.layer4[2].bn2.weight.size(0)),
                             ('glove_embeddings', glove_dim),
                             ('logits', num_classes)])
idx_to_layer = inverse_map(layer_to_idx)

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
        knn.fit(class_emb_vecs)
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

start = time.time()

def merge_and_generate_labels(X_pos, X_neg):
    """
    merge positve and nagative artifact and generate labels
    :param X_pos: positive samples
    :param X_neg: negative samples
    :return: X: merged samples, 2D ndarray
             y: generated labels (0/1): 2D ndarray same size as X
    """
    X_pos = np.asarray(X_pos, dtype=np.float32)
    logger.info("X_pos: {}".format(X_pos.shape))
    X_pos = X_pos.reshape((X_pos.shape[0], -1))

    X_neg = np.asarray(X_neg, dtype=np.float32)
    logger.info("X_neg: {}".format(X_neg.shape))
    X_neg = X_neg.reshape((X_neg.shape[0], -1))

    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    y = y.reshape((X.shape[0], 1))

    return X, y

def sample_estimator():
    num_output = len(layer_to_idx)
    feature_list = np.zeros(num_output, dtype=np.int32)  # indicates the number of features in every layer
    num_sample_per_class = np.zeros(num_classes, dtype=np.int32)  # how many samples are per class
    net.eval()

    for layer_index in range(num_output):
        feature_list[layer_index] = layer_to_size[idx_to_layer[layer_index]]

    list_features = []  # list_features[<layer>][<label>] is a list that holds the features in a specific layer of a specific label
    # is it basically list_features[<num_of_layer>][<num_of_label>] = List

    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append([])
        list_features.append(temp_list)

    for data, target in train_loader:
        data = data.cuda()
        data.requires_grad = True
        outputs = net(data)
        out_features = []
        for layer in layer_to_idx:
            out_features.append(outputs[layer])

        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)

        # construct the sample matrix
        for i in range(data.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] \
                        = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                    out_count += 1
            num_sample_per_class[label] += 1


    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1

    precision = []
    group_lasso = EmpiricalCovariance(assume_centered=False)
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)

        # find inverse
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        precision.append(temp_precision)

    return sample_class_mean, precision

def get_Mahalanobis_score_adv(net, X, y, sample_mean, precision, layer, magnitude):
    '''
    Compute the proposed Mahalanobis confidence score on adversarial samples
    return: Mahalanobis score from layer_index
    '''
    net.eval()
    torch.autograd.set_detect_anomaly(True)
    layer_index = layer_to_idx[layer]
    # converting to tensors:
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    Mahalanobis = []
    num_batch = int(np.ceil(X.shape[0] / batch_size))

    for m in range(num_batch):
        # Batch indexes
        begin, end = (m * batch_size, min((m + 1) * batch_size, X.shape[0]))
        data = X[begin:end].cuda()
        target = y[begin:end].cuda()
        data.requires_grad = True

        out_features = net(data)[layer]
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)

        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1, 1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)

        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - batch_sample_mean
        pure_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()

        if args.use_raw_grads:
            gradient = data.grad
        else:
            gradient = torch.ge(data.grad, 0)
            gradient = (gradient.float() - 0.5) * 2

        # scale hyper params given from the official deep_Mahalanobis_detector repo:
        RED_SCALE   = 0.2023 * args.rgb_scale
        GREEN_SCALE = 0.1994 * args.rgb_scale
        BLUE_SCALE  = 0.2010 * args.rgb_scale
        gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / RED_SCALE)
        gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / GREEN_SCALE)
        gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / BLUE_SCALE)
        tempInputs = torch.add(data, gradient, alpha=-magnitude)

        with torch.no_grad():
            noise_out_features = net(tempInputs)[layer]
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1, 1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        Mahalanobis.extend(noise_gaussian_score.cpu().numpy())

    return Mahalanobis

def get_Mahalanobis(X, X_noisy, X_adv, y, magnitude, sample_mean, precision, set):
    first_pass = True
    for layer in layer_to_idx.keys():
        logger.info('Calculating Mahalanobis characteristics for set {}, for layer {}'.format(set, layer))

        # val
        M_in = get_Mahalanobis_score_adv(net, X, y, sample_mean, precision, layer, magnitude)
        M_in = np.asarray(M_in, dtype=np.float32)

        M_out = get_Mahalanobis_score_adv(net, X_adv, y, sample_mean, precision, layer, magnitude)
        M_out = np.asarray(M_out, dtype=np.float32)

        M_noisy = get_Mahalanobis_score_adv(net, X_noisy, y, sample_mean, precision, layer, magnitude)
        M_noisy = np.asarray(M_noisy, dtype=np.float32)

        if first_pass:
            Mahalanobis_in    = M_in.reshape((M_in.shape[0], -1))
            Mahalanobis_out   = M_out.reshape((M_out.shape[0], -1))
            Mahalanobis_noisy = M_noisy.reshape((M_noisy.shape[0], -1))
            first_pass = False
        else:
            Mahalanobis_in    = np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)
            Mahalanobis_out   = np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)
            Mahalanobis_noisy = np.concatenate((Mahalanobis_noisy, M_noisy.reshape((M_noisy.shape[0], -1))), axis=1)

    if args.include_noise:
        Mahalanobis_neg = np.concatenate((Mahalanobis_in, Mahalanobis_noisy))
    else:
        Mahalanobis_neg = Mahalanobis_in
    Mahalanobis_pos = Mahalanobis_out
    characteristics, labels = merge_and_generate_labels(Mahalanobis_pos, Mahalanobis_neg)

    return characteristics, labels


if args.detect_method == 'mahalanobis':
    logger.info('get sample mean and covariance of the training set...')
    sample_mean, precision = sample_estimator()

    logger.info('Done calculating: sample_mean, precision.')

    if args.magnitude == -1:
        magnitude_vec = [0.00000001, 0.00000003, 0.0000001, 0.0000003, 0.000001, 0.000003, 0.00001, 0.00003,
                         0.0001, 0.0003, 0.001, 0.003, 0.01]
    else:
        magnitude_vec = [args.magnitude]

    for magnitude in magnitude_vec:
        logger.info('Extracting Mahalanobis characteristics for magnitude={}'.format(magnitude))

        # for val set
        characteristics, label = get_Mahalanobis(X_val, X_noisy_val, X_adv_val, y_val, magnitude, sample_mean, precision, 'train')
        logger.info("Mahalanobis train: [characteristic shape: {}, label shape: {}".format(characteristics.shape, label.shape))
        file_name = os.path.join(DUMP_DIR, 'magnitude_{}_train.npy'.format(magnitude))
        data = np.concatenate((characteristics, label), axis=1)
        np.save(file_name, data)
        end_val = time.time()
        logger.info('total feature extraction time for val: {} sec'.format(end_val - start))

        # for test set
        characteristics, label = get_Mahalanobis(X_test, X_noisy_test, X_adv_test, y_test, magnitude, sample_mean, precision, 'test')
        logger.info("Mahalanobis test: [characteristic shape: {}, label shape: {}".format(characteristics.shape, label.shape))
        file_name = os.path.join(DUMP_DIR, 'magnitude_{}_test.npy'.format(magnitude))
        data = np.concatenate((characteristics, label), axis=1)
        np.save(file_name, data)
        end_test = time.time()
        logger.info('total feature extraction time for test: {} sec'.format(end_test - end_val))

logger.handlers[0].flush()
exit(0)

# debug:
# clipping
x_clipped = torch.clip(x, 0.0, 1.0)
x_img = convert_tensor_to_image(x_clipped.detach().cpu().numpy())
for i in range(0, 5):
    plt.imshow(x_img[i])
    plt.show()
