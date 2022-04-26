"""Print an entire experiment line (test acc + all attack acc) to easily dump into the excel sheet"""
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

parser = argparse.ArgumentParser(description='Print experiment line')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/mi/debug', type=str, help='checkpoint dir')
parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')
args = parser.parse_args()

ATTACKS_DIRS = ['gap', 'black_box', 'boundary_distance', 'self_influence']
CHECKPOINT_DIR = '/data/gilad/logs/mi/tiny_imagenet/densenet/relu/s_10k_w_aug'  # args.checkpoint_dir

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

def get_stats_from_line(attack_dir, line: str):
    # data[attack_dir]['member_acc'] = np.round(float(line.split('member acc: ')[1].split(',')[0]), 4)
    # data[attack_dir]['non_member_acc'] = np.round(float(line.split('non-member acc: ')[1].split(',')[0]), 4)
    # data[attack_dir]['balanced_acc'] = np.round(float(line.split('balanced acc: ')[1].split(',')[0]), 4)
    # data[attack_dir]['member_p'] = np.round(float(line.split('precision/recall(member): ')[1].split('/')[0]), 4)
    # data[attack_dir]['member_r'] = np.round(float(line.split('precision/recall(member): ')[1].split('/')[1].split(',')[0]), 4)
    # data[attack_dir]['non_member_p'] = np.round(float(line.split('precision/recall(non-member): ')[1].split('/')[0]), 4)
    # data[attack_dir]['non_member_r'] = np.round(float(line.split('precision/recall(non-member): ')[1].split('/')[1].split(',')[0]), 4)
    data[attack_dir]['member_acc'] = float(line.split('member acc: ')[1].split(',')[0])
    data[attack_dir]['non_member_acc'] = float(line.split('non-member acc: ')[1].split(',')[0])
    data[attack_dir]['balanced_acc'] = float(line.split('balanced acc: ')[1].split(',')[0])
    data[attack_dir]['member_p'] = float(line.split('precision/recall(member): ')[1].split('/')[0])
    data[attack_dir]['member_r'] = float(line.split('precision/recall(member): ')[1].split('/')[1].split(',')[0])
    data[attack_dir]['non_member_p'] = float(line.split('precision/recall(non-member): ')[1].split('/')[0])
    data[attack_dir]['non_member_r'] = float(line.split('precision/recall(non-member): ')[1].split('/')[1].split(',')[0])

def print_to_excel(data):
    # print('{:2f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f}  {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f}'.format(
    print('{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} '.format(
        data['test_acc'],
        data['gap']['member_acc'], data['gap']['member_p'], data['gap']['member_r'],
        data['gap']['non_member_acc'], data['gap']['non_member_p'], data['gap']['non_member_r'], data['gap']['balanced_acc'],
        data['black_box']['member_acc'], data['black_box']['member_p'], data['black_box']['member_r'],
        data['black_box']['non_member_acc'], data['black_box']['non_member_p'], data['black_box']['non_member_r'], data['black_box']['balanced_acc'],
        data['boundary_distance']['member_acc'], data['boundary_distance']['member_p'], data['boundary_distance']['member_r'],
        data['boundary_distance']['non_member_acc'], data['boundary_distance']['non_member_p'], data['boundary_distance']['non_member_r'], data['boundary_distance']['balanced_acc'],
        data['self_influence']['member_acc'], data['self_influence']['member_p'], data['self_influence']['member_r'],
        data['self_influence']['non_member_acc'], data['self_influence']['non_member_p'], data['self_influence']['non_member_r'], data['self_influence']['balanced_acc']
    ))

data = dict()
# getting test acc
file = open(get_log(CHECKPOINT_DIR), 'r')
last_line = file.readlines()[-1]
data['test_acc'] = np.round(float(last_line.split('acc=')[-1].split('\n')[0]), 2)

for attack_dir in ATTACKS_DIRS:
    data[attack_dir] = {}
    file = get_log(os.path.join(CHECKPOINT_DIR, attack_dir))
    line = get_relevant_line_from_log(file)
    get_stats_from_line(attack_dir, line)

print_to_excel(data)

