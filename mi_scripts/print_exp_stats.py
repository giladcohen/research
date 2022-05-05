"""Print experiments' statistics over multiple runs (test acc + inference time)"""
import numpy as np
import os
import argparse
import sys
import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_style("whitegrid")
# np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

parser = argparse.ArgumentParser(description='Print experiments statistics over multiple runs')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/mi/tiny_imagenet/resnet18/relu/s_25k_w_aug', type=str, help='checkpoint dir')
parser.add_argument('--exp_name', default='self_influence_sfast_rec_dep_8_r_8', type=str, help='checkpoint dir')
parser.add_argument('--num_takes', default=5, type=int, help='number of takes for each experiment')
parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')
args = parser.parse_args()

EXP_DIR = os.path.join(args.checkpoint_dir, args.exp_name)
EXP_DIR_VEC = []
for i in range(args.num_takes):
    EXP_DIR_VEC.append(EXP_DIR + '_take' + str(i + 1))

def get_log(dir_path: str):
    path = os.path.join(dir_path, 'log.log')
    return path

def get_relevant_line_from_log(log: str):
    ret = None
    with open(log, 'r') as f:
        for line in f:
            if 'member acc:' in line:
                ret = line
    assert ret is not None
    return ret

def get_stats_from_line(line: str, take: int):
    member_acc = float(line.split('member acc: ')[1].split(',')[0])
    non_member_acc = float(line.split('non-member acc: ')[1].split(',')[0])
    balanced_acc = float(line.split('balanced acc: ')[1].split(',')[0])
    member_p = float(line.split('precision/recall(member): ')[1].split('/')[0])
    member_r = float(line.split('precision/recall(member): ')[1].split('/')[1].split(',')[0])
    non_member_p = float(line.split('precision/recall(non-member): ')[1].split('/')[0])
    non_member_r = float(line.split('precision/recall(non-member): ')[1].split('/')[1].split(',')[0])
    data['member_acc'].append(member_acc)
    data['non_member_acc'].append(non_member_acc)
    data['balanced_acc'].append(balanced_acc)
    data['member_p'].append(member_p)
    data['member_r'].append(member_r)
    data['non_member_p'].append(non_member_p)
    data['non_member_r'].append(non_member_r)

def print_to_excel(data):
    # print('{:2f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f}  {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f}'.format(
    print('{} {} {} {} {} {} {} {} {} {} {} {} {} {}'.format(
        data['mean']['member_acc'], data['std']['member_acc'],
        data['mean']['member_p'], data['std']['member_p'],
        data['mean']['member_r'], data['std']['member_r'],
        data['mean']['non_member_acc'], data['std']['non_member_acc'],
        data['mean']['non_member_p'], data['std']['non_member_p'],
        data['mean']['non_member_r'], data['std']['non_member_r'],
        data['mean']['balanced_acc'], data['std']['balanced_acc']
    ))


data = dict()
for i in range(args.num_takes):
    if i == 0:
        data['member_acc'] = []
        data['non_member_acc'] = []
        data['balanced_acc'] = []
        data['member_p'] = []
        data['member_r'] = []
        data['non_member_p'] = []
        data['non_member_r'] = []
    file = get_log(EXP_DIR_VEC[i])
    line = get_relevant_line_from_log(file)
    get_stats_from_line(line, i + 1)

# get mean and std
data['mean'] = {}
data['std'] = {}
for param in ['member_acc', 'non_member_acc', 'balanced_acc', 'member_p', 'member_r', 'non_member_p', 'non_member_r']:
    data['mean'][param] = np.mean(data[param])
    data['std'][param] = np.std(data[param])

print_to_excel(data)
