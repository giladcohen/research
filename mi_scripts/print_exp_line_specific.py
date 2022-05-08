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
parser.add_argument('--attack_dir', default='/data/gilad/logs/mi/tiny_imagenet/densenet/relu/s_1k_w_aug/adaptive/self_influence_adaptive_rec_dep_8_r_8', type=str, help='checkpoint dir')
parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')
args = parser.parse_args()

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

def get_stats_from_line(line: str):
    data['member_acc'] = float(line.split('member acc: ')[1].split(',')[0])
    data['non_member_acc'] = float(line.split('non-member acc: ')[1].split(',')[0])
    data['balanced_acc'] = float(line.split('balanced acc: ')[1].split(',')[0])
    data['member_p'] = float(line.split('precision/recall(member): ')[1].split('/')[0])
    data['member_r'] = float(line.split('precision/recall(member): ')[1].split('/')[1].split(',')[0])
    data['non_member_p'] = float(line.split('precision/recall(non-member): ')[1].split('/')[0])
    data['non_member_r'] = float(line.split('precision/recall(non-member): ')[1].split('/')[1].split(',')[0])

def print_to_excel(data):
    print('{} {} {} {} {} {} {}'.format(
        data['member_acc'], data['member_p'], data['member_r'],
        data['non_member_acc'], data['non_member_p'], data['non_member_r'], data['balanced_acc'],
    ))

data = dict()
file = get_log(args.attack_dir)
line = get_relevant_line_from_log(file)
get_stats_from_line(line)
print_to_excel(data)
