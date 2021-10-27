"""
Plot accuracy and adversarial accuracy for: CIFAR10
for: resnet34
for methods: simple/ensemble/TRADES/TTA+RF
database is defined as: data[dataset][arch][attack]. attack='' means normal (unattacked) test samples.
"""

import os
import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd
from typing import Dict
import seaborn as sns
sns.set_style("whitegrid")
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
CHECKPOINT_ROOT = '/data/gilad/logs/glove_emb'
datasets = ['cifar10']
nets = ['resnet34']
# emb_selections = ['glove', 'random', 'orthogonal', 'farthest_points']
# metrics = ['L1', 'L2', 'Linf', 'cosine']
emb_selections = ['glove']
metrics = ['cosine']
attacks = ['normal', 'fgsm_L1', 'fgsm_L2', 'fgsm_Linf', 'fgsm_cosine', 'pgd_L1', 'pgd_L2', 'pgd_Linf', 'pgd_cosine']
evals = ['knn_p1', 'knn_p2', 'knn_pinf', 'cosine']
data = {}

def get_log(dataset, net, emb_selection, metric, attack, eval):
    path = os.path.join(CHECKPOINT_ROOT, dataset, net, emb_selection + '_adv_robust',
                        emb_selection + '_emb_' + metric + '_e_' + metric + '_loss_beta_6', attack, eval)
    path = os.path.join(path, 'log.log')
    return path

def get_acc_from_log(log: str):
    ret = None
    with open(log, 'r') as f:
        for line in f:
            if 'INFO Test accuracy:' in line:
                ret = float(line.split('accuracy: ')[1].split('%')[0])
    assert ret is not None
    ret = np.round(ret, 2)
    return ret

def print_metrics_for_attack(ad):
    print('L1: {}, L2: {}, Linf: {}, cosine: {}'.format(ad['knn_p1'], ad['knn_p2'], ad['knn_pinf'], ad['cosine']))


for dataset in datasets:
    data[dataset] = {}
    for net in nets:
        data[dataset][net] = {}
        for emb_selection in emb_selections:
            data[dataset][net][emb_selection] = {}
            for metric in metrics:
                data[dataset][net][emb_selection][metric] = {}
                for attack in attacks:
                    data[dataset][net][emb_selection][metric][attack] = {}
                    for eval in evals:
                        log = get_log(dataset, net, emb_selection, metric, attack, eval)
                        data[dataset][net][emb_selection][metric][attack][eval] = get_acc_from_log(log)
                    # print('For dataset: {}, train_dir: {}, attack: {} we got:'.format(dataset, train_dir, attack))
                    # print('for train_dir: {}, attack: {} we got:'.format(train_dir, attack))
                    print('for attack {} we got:'.format(os.path.dirname(os.path.dirname(log))))
                    print_metrics_for_attack(data[dataset][net][emb_selection][metric][attack])
