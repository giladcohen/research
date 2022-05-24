"""Plotting the MI attack results for CIFAR10"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

datasets = ['cifar10', 'cifar100', 'tiny_imagenet']
target_models = ['1', '2', '3', '4', '5', '6', '7']
attacks  = ['Gap', 'Black-box', 'Boundary dist', 'SIF']

dataset = datasets[1]
n_groups = len(target_models)

attack_score_dict = {dataset: {
    '1': {'Gap': 0.52,    'Black-box': 0.42,    'Boundary dist': 0.5,   'SIF': 0.53},       # 100
    '2': {'Gap': 0.969,   'Black-box': 0.798,   'Boundary dist': 0.98,  'SIF': 0.978},   # 1k
    '3': {'Gap': 0.6712,  'Black-box': 0.525,   'Boundary dist': 0.656, 'SIF': 0.6712},  # 5k
    '4': {'Gap': 0.9139,  'Black-box': 0.903,   'Boundary dist': 0.937, 'SIF': 0.9486},  # 10k
    '5': {'Gap': 0.8694,  'Black-box': 0.869266666666666,  'Boundary dist': 0.909, 'SIF': 0.941066666666666},  # 15k
    '6': {'Gap': 0.86575, 'Black-box': 0.897,   'Boundary dist': 0.903, 'SIF': 0.94475},  # 20k
    '7': {'Gap': 0.83384, 'Black-box': 0.87248, 'Boundary dist': 0.873, 'SIF': 0.92424},  # 25k
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
                 hatch='-',
                 label='Black-box')

values3 = [attack_score_dict[dataset]['1']['Boundary dist'], attack_score_dict[dataset]['2']['Boundary dist'],
           attack_score_dict[dataset]['3']['Boundary dist'], attack_score_dict[dataset]['4']['Boundary dist'],
           attack_score_dict[dataset]['5']['Boundary dist'], attack_score_dict[dataset]['6']['Boundary dist'],
           attack_score_dict[dataset]['7']['Boundary dist']]
rects3 = plt.bar(index + 3*bar_width, values3, bar_width,
                 alpha=opacity,
                 color='green',
                 edgecolor='black',
                 hatch='.',
                 label='Boundary dist')

values4 = [attack_score_dict[dataset]['1']['SIF'], attack_score_dict[dataset]['2']['SIF'],
           attack_score_dict[dataset]['3']['SIF'], attack_score_dict[dataset]['4']['SIF'],
           attack_score_dict[dataset]['5']['SIF'], attack_score_dict[dataset]['6']['SIF'],
           attack_score_dict[dataset]['7']['SIF']]
rects4 = plt.bar(index + 4*bar_width, values4, bar_width,
                 alpha=opacity,
                 color='red',
                 edgecolor='black',
                 hatch='/',
                 label='SIF (ours)')

plt.xlabel('Target Model $\mathcal{M}$')
plt.ylabel('Balanced Acc')
plt.ylim(bottom=0.4, top=1.0)
plt.xticks(index + 2.5*bar_width, ('1', '2', '3', '4', '5', '6', '7'))
plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# plt.legend((rects1, rects2, rects3, rects4), ('Gap', 'Black-box', 'Boundary dist', 'SIF (ours)'),
#            loc=(0.63, 0.77), ncol=1, fancybox=True, prop={'size': 10})
plt.title('CIFAR-100')
plt.tight_layout()
plt.savefig('cifar100_alexnet_attack_scores.png', dpi=350)
plt.show()

