import matplotlib
import platform
# Force matplotlib to not use any Xwindows backend.
if platform.system() == 'Linux':
    matplotlib.use('Agg')
import os
import numpy as np
from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.decomposition import PCA
import argparse
import json
import glob
from research.adv_detection_utils import random_split, block_split, train_lr, compute_roc
from research.utils import boolean_string, load_characteristics

parser = argparse.ArgumentParser(description='Evaluating robustness score')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/glove_emb/cifar10/resnet34_glove_p2', type=str, help='checkpoint dir')
parser.add_argument('--seen_attack_dir', default='', type=str, help='Seen attack when training detector')
parser.add_argument('--attack_dir', default='fgsm_L2', type=str, help='attack directory')
parser.add_argument('--detect_method', default='mahalanobis', type=str, help='lid/mahalanobis/dknn')
parser.add_argument('--dump_dir', default='mahalanobis', type=str, help='dump dir for logs and characteristics')

# for lid/dknn
parser.add_argument('--k_nearest', default=-1, type=int, help='number of nearest neighbors to use for LID/dknn detection')

# for mahalanobis
parser.add_argument('--magnitude', default=-1, type=float, help='magnitude for mahalanobis detection')
parser.add_argument('--rgb_scale', default=1, type=float, help='scale for mahalanobis')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

ATTACK_DIR = os.path.join(args.checkpoint_dir, args.attack_dir)
DUMP_DIR = os.path.join(ATTACK_DIR, args.dump_dir)
if args.seen_attack_dir != '':
    SEEN_ATTACK_DIR = os.path.join(args.checkpoint_dir, args.seen_attack_dir)
else:
    SEEN_ATTACK_DIR = ATTACK_DIR
SEEN_DUMP_DIR = os.path.join(SEEN_ATTACK_DIR, args.dump_dir)
with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)
with open(os.path.join(SEEN_ATTACK_DIR, 'attack_args.txt'), 'r') as f:
    seen_attack_args = json.load(f)
with open(os.path.join(ATTACK_DIR, 'attack_args.txt'), 'r') as f:
    attack_args = json.load(f)

dataset = train_args['dataset']

print("Testing on [dataset: %s, seen_attack_dir: %s, test_attack: %s, detector: %s]:"
      % (dataset, args.seen_attack_dir if args.seen_attack_dir != '' else args.attack_dir, args.attack_dir, args.detect_method))

# collect characteristics for all parameters in the test folder:
def collect_params(dir_path):
    files = []
    files = glob.glob(os.path.join(dir_path, '*_test.npy'))
    files = [os.path.basename(f) for f in files]
    params = [name.split('_')[1] for name in files]
    return params


param_vec = collect_params(DUMP_DIR)
if args.detect_method == 'mahalanobis':
    param_name = 'magnitude'
elif args.detect_method == 'lid':
    param_name = 'k_nearest'
else:
    raise AssertionError('Unknown detection {}'.format(args.detect_method))

best_auc = 0.0
best_accuracy = 0.0
best_precision = 0.0
best_recall = 0.0
best_param = None
for param in param_vec:
    train_characteristics_file = os.path.join(SEEN_DUMP_DIR, '{}_{}_train.npy'.format(param_name, param))
    test_characteristics_file  = os.path.join(DUMP_DIR, '{}_{}_test.npy'.format(param_name, param))

    print("Loading train attack: {}\nTraining file: {}\nTesting file: {}".
          format(args.attack_dir, train_characteristics_file, test_characteristics_file))
    X_train, Y_train = load_characteristics(train_characteristics_file)
    X_test, Y_test   = load_characteristics(test_characteristics_file)

    scaler  = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)

    print("Train data size: ", X_train.shape)
    print("Test data size: ", X_test.shape)

    # Build detector
    # print("LR Detector on [dataset: %s, test_attack: %s, detector: %s]:" % (dataset, args.attack_dir, args.detect_method))
    lr = train_lr(X_train, Y_train)

    # Evaluate detector
    y_pred       = lr.predict_proba(X_test)[:, 1]
    y_label_pred = lr.predict(X_test)

    # AUC
    _, _, auc_score = compute_roc(Y_test, y_pred, plot=False)
    precision = precision_score(Y_test, y_label_pred)
    recall    = recall_score(Y_test, y_label_pred)
    acc       = accuracy_score(Y_test, y_label_pred)

    if auc_score > best_auc:
        best_auc = auc_score
        best_accuracy = acc
        best_precision = precision
        best_recall = recall
        best_param = param

print('Detector ROC-AUC score: {}, accuracy: {}, precision: {}, recall: {}, obtained for {}: {} '.
      format(best_auc, best_accuracy, best_precision, best_recall, param_name, best_param))
