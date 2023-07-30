"""Plot ablation study of the rec_dep value for CIFAR10, CIFAR100, and Tiny ImageNet"""

import os
import matplotlib
import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style("whitegrid")
sns.set(rc={'figure.figsize': (5, 5)})

def get_relevant_line_from_log(log: str):
    ret = None
    with open(log, 'r') as f:
        for line in f:
            if 'member acc:' in line:
                ret = line
    assert ret is not None
    return ret

def get_acc_from_line(line: str):
    balanced_acc = float(line.split('balanced acc: ')[1].split(',')[0])
    return balanced_acc

num_experiments = 5
datasets = ['cifar10', 'cifar100', 'tiny_imagenet']
CHECKPOINT_ROOT = '/data/gilad/logs/mi'
rec_dep_vec = [1, 2, 4, 8, 16, 32, 64, 128]
rec_dep_ext_vec = []
data_ext = {}
for size in rec_dep_vec:
    rec_dep_ext_vec.extend([size] * num_experiments)

data = {}
for dataset in datasets:
    data[dataset] = {}
    for rec_dep_size in rec_dep_vec:
        data[dataset][rec_dep_size] = []
        for n in range(1, num_experiments + 1):
            file = os.path.join(CHECKPOINT_ROOT, dataset, 'resnet18', 'relu', 's_25k_w_aug', 'self_influence_sfast_rec_dep_{}_r_1_take{}'.format(rec_dep_size, n), 'log.log')
            line = get_relevant_line_from_log(file)
            acc = get_acc_from_line(line)
            data[dataset][rec_dep_size].append(acc)
        data[dataset][rec_dep_size] = np.asarray(data[dataset][rec_dep_size])

d = {'rec_dep_size': rec_dep_ext_vec}
for dataset in datasets:
    if dataset == 'cifar10':
        dataset_str = 'CIFAR-10'
    elif dataset == 'cifar100':
        dataset_str = 'CIFAR-100'
    else:
        dataset_str = 'Tiny ImageNet'
    data_ext[dataset] = []
    for size in rec_dep_vec:
        data_ext[dataset].extend(data[dataset][size])
        d.update({dataset_str: data_ext[dataset]})

df = pd.DataFrame(d)
g = sns.lineplot(x='rec_dep_size', y='value', hue='variable', data=pd.melt(df, ['rec_dep_size']), ci='sd',
                 palette='husl', style='variable')
g.set(xscale='log', xlabel='Recursion depth ($d$)', ylabel='Balanced Acc')
g.legend_.set_title('')
g.set_xticks(rec_dep_vec)
g.set_xticklabels(rec_dep_vec)
plt.tight_layout()
plt.legend(loc=(0.05, 0.3))
plt.savefig('rec_dep_ablation.png', dpi=350)
plt.show()
