"""Plotting the MI attack results for CIFAR10"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

datasets = ['cifar10', 'cifar100', 'svhn']
target_models = ['1', '2', '3', '4', '5', '6', '7']
attacks  = ['Gap', 'Black-box', 'Boundary dist', 'SIF']

dataset = datasets[1]
n_groups = len(target_models)

attack_score_dict = {dataset: {
    '1': {'Gap': 1.0,     'Black-box': 0.46,    'Boundary dist': 1.0,   'SIF': 1},       # 100
    '2': {'Gap': 0.923,   'Black-box': 0.926,   'Boundary dist': 0.977, 'SIF': 0.996},   # 1k
    '3': {'Gap': 0.8816,  'Black-box': 0.9674,  'Boundary dist': 0.963, 'SIF': 0.9904},  # 5k
    '4': {'Gap': 0.8458,  'Black-box': 0.9566,  'Boundary dist': 0.947, 'SIF': 0.9801},  # 10k
    '5': {'Gap': 0.8006,  'Black-box': 0.953266666666666,  'Boundary dist': 0.918, 'SIF': 0.976266666666666},  # 15k
    '6': {'Gap': 0.76725, 'Black-box': 0.94175, 'Boundary dist': 0.913, 'SIF': 0.9661},  # 20k
    '7': {'Gap': 0.76196, 'Black-box': 0.92728, 'Boundary dist': 0.888, 'SIF': 0.94924},  # 25k
}}

fig, ax = plt.subplots(figsize=(5, 5))
index = np.arange(n_groups)
bar_width = 0.15
opacity = 0.4

# attack rectangle: all of Gap
values1 = [attack_score_dict[dataset]['1']['Gap'], attack_score_dict[dataset]['2']['Gap'],
           attack_score_dict[dataset]['3']['Gap'], attack_score_dict[dataset]['4']['Gap'],
           attack_score_dict[dataset]['5']['Gap'], attack_score_dict[dataset]['6']['Gap'],
           attack_score_dict[dataset]['7']['Gap']]
rects1 = plt.bar(index + bar_width, values1, bar_width,
                 alpha=opacity,
                 color='black',
                 edgecolor='black',
                 label='Gap')

values2 = [attack_score_dict[dataset]['1']['Black-box'], attack_score_dict[dataset]['2']['Black-box'],
           attack_score_dict[dataset]['3']['Black-box'], attack_score_dict[dataset]['4']['Black-box'],
           attack_score_dict[dataset]['5']['Black-box'], attack_score_dict[dataset]['6']['Black-box'],
           attack_score_dict[dataset]['7']['Black-box']]
rects2 = plt.bar(index + 2*bar_width, values2, bar_width,
                 alpha=opacity,
                 color='blue',
                 edgecolor='black',
                 label='Black-box')

values3 = [attack_score_dict[dataset]['1']['Boundary dist'], attack_score_dict[dataset]['2']['Boundary dist'],
           attack_score_dict[dataset]['3']['Boundary dist'], attack_score_dict[dataset]['4']['Boundary dist'],
           attack_score_dict[dataset]['5']['Boundary dist'], attack_score_dict[dataset]['6']['Boundary dist'],
           attack_score_dict[dataset]['7']['Boundary dist']]
rects3 = plt.bar(index + 3*bar_width, values3, bar_width,
                 alpha=opacity,
                 color='green',
                 edgecolor='black',
                 label='Boundary dist')

values4 = [attack_score_dict[dataset]['1']['SIF'], attack_score_dict[dataset]['2']['SIF'],
           attack_score_dict[dataset]['3']['SIF'], attack_score_dict[dataset]['4']['SIF'],
           attack_score_dict[dataset]['5']['SIF'], attack_score_dict[dataset]['6']['SIF'],
           attack_score_dict[dataset]['7']['SIF']]
rects4 = plt.bar(index + 4*bar_width, values4, bar_width,
                 alpha=opacity,
                 color='red',
                 edgecolor='black',
                 label='SIF (ours)')

plt.xlabel('Target Model $\mathcal{M}$')
plt.ylabel('Balanced Acc')
plt.ylim(bottom=0.44, top=1.04)
plt.xticks(index + 2.5*bar_width, ('1', '2', '3', '4', '5', '6', '7'))
plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# plt.legend((rects1, rects2, rects3, rects4), ('Gap', 'Black-box', 'Boundary dist', 'SIF (ours)'),
#            loc=(0.63, 0.77), ncol=1, fancybox=True, prop={'size': 10})
plt.tight_layout()
# plt.show()
plt.savefig('cifar00_attack_scores.png', dpi=350)

