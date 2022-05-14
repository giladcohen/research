"""Plot SIF distribution of both 'members' and 'non-members'"""
import numpy as np
import os
import argparse
import sys
import logging
from matplotlib import rcParams
import matplotlib.pyplot as plt
# figure size in inches
rcParams['figure.figsize'] = 4, 4
import pandas as pd
import seaborn as sns
sns.set_style("whitegrid")
# np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

parser = argparse.ArgumentParser(description='Plot histogram of SIF values')
parser.add_argument('--attack_dir', default='/Users/giladcohen/logs/cifar10/resnet18/relu/s_25k_wo_aug/self_influence_v2', type=str, help='attack dir')
parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')
args = parser.parse_args()

self_influences_member_train = np.load(os.path.join(args.attack_dir, 'self_influences_member_train.npy'))
self_influences_non_member_train = np.load(os.path.join(args.attack_dir, 'self_influences_non_member_train.npy'))
self_influences_member_test = np.load(os.path.join(args.attack_dir, 'self_influences_member_test.npy'))
self_influences_non_member_test = np.load(os.path.join(args.attack_dir, 'self_influences_non_member_test.npy'))

sif_members = np.concatenate((self_influences_member_train, self_influences_member_test))
sif_non_members = np.concatenate((self_influences_non_member_train, self_influences_non_member_test))
mem_df = pd.DataFrame(sif_members, columns=['SIF value'])
non_mem_df = pd.DataFrame(sif_non_members, columns=['SIF value'])


# CIFAR10, M7
sns.histplot(data=mem_df, x='SIF value', bins=30, binrange=[-0.0005, 0.000005], color='orange', log_scale=[False, True])
plt.legend(labels=['Members'], loc='upper left')  # prop={'size': 12}
plt.savefig('sif_hist_members.png', dpi=350)
plt.show()

sns.histplot(data=non_mem_df, x='SIF value', bins=30, color='blue', log_scale=[False, True], legend=True)
plt.xticks([-800000, -600000, -400000, -200000, 0, 200000, 400000, 600000], ['-800k', '-600k', '-400k', '-200k', '0', '200k', '400k', '600k'])
plt.legend(labels=['Non-members'], loc='upper left')
plt.savefig('sif_hist_non_members.png', dpi=350)
plt.show()

# joint plot
sns.histplot(data=non_mem_df, x='SIF value', bins=30, binrange=[-0.0005, 0.000005], color='blue', log_scale=[False, True], legend=True)
sns.histplot(data=mem_df, x='SIF value', bins=30, binrange=[-0.0005, 0.000005], color='red', log_scale=[False, True], legend=True, fill=False)
plt.legend(labels=['Members', 'Non-members'], loc='upper left')
ax = plt.gca()
leg = ax.get_legend()
leg.legendHandles[0].set_facecolor('white')
leg.legendHandles[0].set_edgecolor('red')
leg.legendHandles[1].set_color('blue')
plt.savefig('sif_hist_joint.png', dpi=350)
plt.show()

# # CIFAR100, M7
# sns.histplot(data=mem_df, x='SIF value', bins=30, binrange=[-0.003, 0.0015], color='orange', log_scale=[False, True])
# plt.legend(labels=['Members'], loc='upper left')  # prop={'size': 12}
# plt.savefig('sif_hist_members.png', dpi=350)
# plt.show()
#
# sns.histplot(data=non_mem_df, x='SIF value', bins=30, binrange=[-40000, 40000], color='blue', log_scale=[False, True], legend=True)
# plt.xticks([-40000, -20000, 0, 20000, 40000], ['-40k', '-20k', '0', '20k', '40k'])
# plt.legend(labels=['Non-members'], loc='upper left')
# plt.savefig('sif_hist_non_members.png', dpi=350)
# plt.show()
#
# # joint plot
# sns.histplot(data=non_mem_df, x='SIF value', bins=30, binrange=[-0.003, 0.0015], color='blue', log_scale=[False, True], legend=True)
# sns.histplot(data=mem_df, x='SIF value', bins=30, binrange=[-0.003, 0.0015], color='red', log_scale=[False, True], legend=True, fill=False)
# plt.legend(labels=['Members', 'Non-members'], loc='upper left')
# ax = plt.gca()
# leg = ax.get_legend()
# leg.legendHandles[0].set_facecolor('white')
# leg.legendHandles[0].set_edgecolor('red')
# leg.legendHandles[1].set_color('blue')
# plt.savefig('sif_hist_joint.png', dpi=350)
# plt.show()

# # CIFAR10, M1
# sns.histplot(data=mem_df, x='SIF value', bins=30, binrange=[-0.003, 0.026], color='orange', log_scale=[False, True])
# plt.legend(labels=['Members'], loc='upper right')  # prop={'size': 12}
# plt.savefig('sif_hist_members.png', dpi=350)
# plt.show()
#
# sns.histplot(data=non_mem_df, x='SIF value', bins=30, binrange=[-40000, 36000], color='blue', log_scale=[False, True], legend=True)
# plt.xticks([-40000, -20000, 0, 20000], ['-40k', '-20k', '0', '20k'])
# plt.legend(labels=['Non-members'], loc='upper right')
# plt.savefig('sif_hist_non_members.png', dpi=350)
# plt.show()
#
# # joint plot
# sns.histplot(data=non_mem_df, x='SIF value', bins=30, binrange=[-0.003, 0.026], color='blue', log_scale=[False, True], legend=True)
# sns.histplot(data=mem_df, x='SIF value', bins=30, binrange=[-0.003, 0.026], color='red', log_scale=[False, True], legend=True, fill=False)
# plt.legend(labels=['Members', 'Non-members'], loc='upper right')
# ax = plt.gca()
# leg = ax.get_legend()
# leg.legendHandles[0].set_facecolor('white')
# leg.legendHandles[0].set_edgecolor('red')
# leg.legendHandles[1].set_color('blue')
# plt.savefig('sif_hist_joint.png', dpi=350)
# plt.show()


# # CIFAR100, M1
# sns.histplot(data=mem_df, x='SIF value', bins=30, binrange=[-0.011, 0.038], color='red', fill=False, log_scale=[False, True])
# plt.legend(labels=['Members'], loc='upper right')  # prop={'size': 12}
# plt.savefig('sif_hist_members.png', dpi=350)
# plt.show()
#
# sns.histplot(data=non_mem_df, x='SIF value', bins=30, color='blue', log_scale=[False, True], legend=True)
# plt.legend(labels=['Non-members'], loc='upper left')
# plt.savefig('sif_hist_non_members.png', dpi=350)
# plt.show()
#
# # joint plot
# sns.histplot(data=non_mem_df, x='SIF value', bins=30, binrange=[-0.011, 0.038], color='blue', log_scale=[False, True], legend=True)
# sns.histplot(data=mem_df, x='SIF value', bins=30, binrange=[-0.011, 0.038], color='red', log_scale=[False, True], legend=True, fill=False)
# plt.legend(labels=['Members', 'Non-members'], loc='upper right')
# ax = plt.gca()
# leg = ax.get_legend()
# leg.legendHandles[0].set_facecolor('white')
# leg.legendHandles[0].set_edgecolor('red')
# leg.legendHandles[1].set_color('blue')
# plt.savefig('sif_hist_joint.png', dpi=350)
# plt.show()
