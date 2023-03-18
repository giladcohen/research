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
    '2': {'Gap': 0.498,   'Black-box': 0.444,   'Boundary dist': 0.498, 'SIF': 0.498},   # 1k
    '3': {'Gap': 0.648,   'Black-box': 0.5402,  'Boundary dist': 0.645, 'SIF': 0.648},  # 5k
    '4': {'Gap': 0.5577,  'Black-box': 0.4964,  'Boundary dist': 0.551, 'SIF': 0.5576},  # 10k
    '5': {'Gap': 0.9628,  'Black-box': 0.941066666666666,  'Boundary dist': 0.974, 'SIF': 0.9912},  # 15k
    '6': {'Gap': 0.62405, 'Black-box': 0.51005,  'Boundary dist': 0.616, 'SIF': 0.6235},  # 20k
    '7': {'Gap': 0.57068, 'Black-box': 0.50204, 'Boundary dist': 0.553, 'SIF': 0.57072},  # 25k
}}

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

plt.xlabel('Target Model $\mathcal{M}$')
plt.ylabel('Balanced Acc')
plt.ylim(bottom=0.4, top=1.002)
plt.xticks(index + 2.5*bar_width, ('2', '3', '4', '5', '6', '7'))
# plt.yticks([0.9, 0.95, 1.0])
# plt.legend((rects1, rects2, rects3, rects4), ('Gap', 'Black-box', 'Boundary dist', 'SIF (ours)'),
#            loc=(0.63, 0.77), ncol=1, fancybox=True, prop={'size': 10})
plt.title('Tiny ImageNet')
plt.tight_layout()
plt.savefig('tiny_imagenet_alexnet_attack_scores.png', dpi=350)
plt.show()

