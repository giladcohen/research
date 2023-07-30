"""Plotting the MI attack results for CIFAR10"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

datasets = ['cifar10', 'cifar100', 'tiny_imagenet']
target_models = ['2', '3', '4', '5', '6', '7']
attacks  = ['Gap', 'Black-box', 'Boundary dist', 'SIF']

dataset = datasets[2]
n_groups = len(target_models)

attack_score_dict = {dataset: {
    '2': {'Gap': 0.985,   'Black-box': 0.85,   'Boundary dist': 0.985, 'SIF': 0.989},   # 1k
    '3': {'Gap': 0.9112,  'Black-box': 0.7266, 'Boundary dist': 0.91,  'SIF': 0.9094},  # 5k
    '4': {'Gap': 0.7836,  'Black-box': 0.595,  'Boundary dist': 0.761, 'SIF': 0.7833},  # 10k
    '5': {'Gap': 0.6824,  'Black-box': 0.533533333333333,  'Boundary dist': 0.668, 'SIF': 0.6822},  # 15k
    '6': {'Gap': 0.6791,  'Black-box': 0.52425, 'Boundary dist': 0.676, 'SIF': 0.67815},  # 20k
    '7': {'Gap': 0.66896, 'Black-box': 0.52848, 'Boundary dist': 0.658, 'SIF': 0.66924},  # 25k
}}

attack_score_adaptive_dict = {dataset: {
    '2': {'Gap': 0.985,   'Black-box': 0.85,   'Boundary dist': 0.985, 'SIF': 0.987},   # 1k
    '3': {'Gap': 0.9112,  'Black-box': 0.7266, 'Boundary dist': 0.91,  'SIF': 0.9102},  # 5k
    '4': {'Gap': 0.7836,  'Black-box': 0.595,  'Boundary dist': 0.761, 'SIF': 0.7866},  # 10k
    '5': {'Gap': 0.6824,  'Black-box': 0.533533333333333,  'Boundary dist': 0.668, 'SIF': 0.6828},  # 15k
    '6': {'Gap': 0.6791,  'Black-box': 0.52425, 'Boundary dist': 0.676, 'SIF': 0.676},  # 20k
    '7': {'Gap': 0.66896, 'Black-box': 0.52848, 'Boundary dist': 0.658, 'SIF': 0.6774},  # 25k
}}

adaptive_boost_dict = {}
for key1 in attack_score_adaptive_dict.keys():
    adaptive_boost_dict[key1] = {}
    for key2 in attack_score_adaptive_dict[key1].keys():
        adaptive_boost_dict[key1][key2] = {}
        for key3 in attack_score_adaptive_dict[key1][key2].keys():
            adaptive_boost_dict[key1][key2][key3] = \
                np.maximum(0.0, attack_score_adaptive_dict[key1][key2][key3] - attack_score_dict[key1][key2][key3])

fig, ax = plt.subplots(figsize=(5, 5))
index = np.arange(n_groups)
bar_width = 0.15
opacity = 0.4

# attack rectangle: all of Gap
values1 = [attack_score_dict[dataset]['2']['Gap'],
           attack_score_dict[dataset]['3']['Gap'], attack_score_dict[dataset]['4']['Gap'],
           attack_score_dict[dataset]['5']['Gap'], attack_score_dict[dataset]['6']['Gap'],
           attack_score_dict[dataset]['7']['Gap']]
rects1 = plt.bar(index + bar_width, values1, bar_width,
                 alpha=opacity,
                 color='black',
                 edgecolor='black',
                 label='Gap')

# attack rectangle: all of Black-box
values2 = [attack_score_dict[dataset]['2']['Black-box'],
           attack_score_dict[dataset]['3']['Black-box'], attack_score_dict[dataset]['4']['Black-box'],
           attack_score_dict[dataset]['5']['Black-box'], attack_score_dict[dataset]['6']['Black-box'],
           attack_score_dict[dataset]['7']['Black-box']]
rects2 = plt.bar(index + 2*bar_width, values2, bar_width,
                 alpha=opacity,
                 color='blue',
                 edgecolor='black',
                 hatch='-',
                 label='Black-box')

# attack rectangle: all of Boundary distance
values3 = [attack_score_dict[dataset]['2']['Boundary dist'],
           attack_score_dict[dataset]['3']['Boundary dist'], attack_score_dict[dataset]['4']['Boundary dist'],
           attack_score_dict[dataset]['5']['Boundary dist'], attack_score_dict[dataset]['6']['Boundary dist'],
           attack_score_dict[dataset]['7']['Boundary dist']]
rects3 = plt.bar(index + 3*bar_width, values3, bar_width,
                 alpha=opacity,
                 color='green',
                 edgecolor='black',
                 hatch='.',
                 label='Boundary dist')

# attack rectangle: all of SIF
values4 = [attack_score_dict[dataset]['2']['SIF'],
           attack_score_dict[dataset]['3']['SIF'], attack_score_dict[dataset]['4']['SIF'],
           attack_score_dict[dataset]['5']['SIF'], attack_score_dict[dataset]['6']['SIF'],
           attack_score_dict[dataset]['7']['SIF']]
rects4 = plt.bar(index + 4*bar_width, values4, bar_width,
                 alpha=opacity,
                 color='red',
                 edgecolor='black',
                 hatch='/',
                 label='SIF (ours)')

values44 = [adaptive_boost_dict[dataset]['2']['SIF'],
            adaptive_boost_dict[dataset]['3']['SIF'], adaptive_boost_dict[dataset]['4']['SIF'],
            adaptive_boost_dict[dataset]['5']['SIF'], adaptive_boost_dict[dataset]['6']['SIF'],
            adaptive_boost_dict[dataset]['7']['SIF']]
rects44 = plt.bar(index + 4*bar_width, values44, bar_width,
                  # alpha=opacity,
                  color='red',
                  edgecolor='black',
                  hatch='/',
                  bottom=values4)

colorless_patch = mpatches.Patch(label='Adaptive', hatch='/', edgecolor='black', facecolor='red')

plt.xlabel('Target Model $\mathcal{M}$')
plt.ylabel('Balanced Acc')
plt.ylim(bottom=0.5, top=1.0)
plt.xticks(index + 2.5*bar_width, ('2', '3', '4', '5', '6', '7'))
# plt.yticks([0.8, 0.9, 1.0])
plt.title('Tiny ImageNet')
# plt.legend((rects1, rects2, rects3, rects4, colorless_patch), ('Gap', 'Black-box', 'Boundary dist', 'SIF (ours)', 'Adaptive'),
#            loc=(0.63, 0.70), ncol=1, fancybox=True, prop={'size': 10})
plt.tight_layout()
plt.savefig('tiny_imagenet_densenet_adaptive_attack_scores.png', dpi=350)
plt.show()
