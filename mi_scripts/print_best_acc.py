"""Print an entire experiment line (test acc + all attack acc) to easily dump into the excel sheet"""
import numpy as np
import torch
import os
import argparse
import sys
import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_style("whitegrid")
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

parser = argparse.ArgumentParser(description='Print best accuracy a the model')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/mi/cifar10/alexnet/relu/s_100_wo_aug', type=str, help='checkpoint dir')
parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')
args = parser.parse_args()

def get_best_acc_train(log: str, best_epoch: int):
    ret = None
    line_str = "INFO Epoch #{} (TRAIN)".format(best_epoch)
    with open(log, 'r') as f:
        for line in f:
            if line_str in line:
                ret = line
    assert ret is not None
    return ret

def get_best_acc_test(log: str):
    with open(log, 'r') as f:
        line = f.readlines()[-1]
    ret = line
    assert "INFO Epoch #400 (TEST)" in ret
    return ret

CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')
global_state = torch.load(CHECKPOINT_PATH)
best_epoch = global_state['epoch'] + 1
best_val = np.round(global_state['best_metric'], 2)

LOG_FILE = os.path.join(args.checkpoint_dir, 'log.log')
train_loss_line = get_best_acc_train(LOG_FILE, best_epoch)
test_loss_line = get_best_acc_test(LOG_FILE)
best_train = np.round(float(train_loss_line.split("\tacc=")[1].split('\n')[0]), 2)
best_test = np.round(float(test_loss_line.split("\tacc=")[1].split('\n')[0]), 2)

print('Train/Val/Test accuracies are:')
# print(str(best_train) + ' & ' + str(best_val) + ' & ' + str(best_test))
print('{} & {} & {}'.format(best_train, best_val, best_test))
