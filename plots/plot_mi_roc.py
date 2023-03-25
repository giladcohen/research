import matplotlib.pyplot as plt
import numpy as np
import os

gap_dir, black_box_dir, self_influence_dir = 'gap', 'black_box', 'self_influence_v2'

# for CIFAR-10, resnet, w.o augmentations
model_dir = '/data/gilad/logs/mi/cifar10/resnet18/relu/s_25k_wo_aug'
gap_fpr = np.load(os.path.join(model_dir, gap_dir, 'fpr.npy'))
gap_tpr = np.load(os.path.join(model_dir, gap_dir, 'tpr.npy'))
gap_thd = np.load(os.path.join(model_dir, gap_dir, 'thresholds.npy'))

black_box_fpr = np.load(os.path.join(model_dir, black_box_dir, 'fpr.npy'))
black_box_tpr = np.load(os.path.join(model_dir, black_box_dir, 'tpr.npy'))
black_box_thd = np.load(os.path.join(model_dir, black_box_dir, 'thresholds.npy'))

self_influence_fpr = np.load(os.path.join(model_dir, self_influence_dir, 'fpr.npy'))
self_influence_tpr = np.load(os.path.join(model_dir, self_influence_dir, 'tpr.npy'))
self_influence_thd = np.load(os.path.join(model_dir, self_influence_dir, 'thresholds.npy'))


