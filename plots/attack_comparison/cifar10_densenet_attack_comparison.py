"""Plotting the MI attack results for CIFAR10"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

datasets = ['cifar10', 'cifar100', 'tiny_imagenet']
target_models = ['1', '2', '3', '4', '5', '6', '7']
attacks  = ['Gap', 'Black-box', 'Boundary dist', 'SIF']

dataset = datasets[0]
n_groups = len(target_models)

attack_score_dict = {dataset: {
    '1': {'Gap': 0.9,     'Black-box': 0.63,    'Boundary dist': 0.87,  'SIF': 0.92},    # 100
    '2': {'Gap': 0.777,   'Black-box': 0.748,   'Boundary dist': 0.87, 'SIF': 0.916},   # 1k
    '3': {'Gap': 0.6754,  'Black-box': 0.7204,  'Boundary dist': 0.784, 'SIF': 0.8038},  # 5k
    '4': {'Gap': 0.6273,  'Black-box': 0.7035,  'Boundary dist': 0.729, 'SIF': 0.7229},  # 10k
    '5': {'Gap': 0.611866666666666,   'Black-box': 0.6806,  'Boundary dist': 0.665, 'SIF': 0.714066666666666},  # 15k
    '6': {'Gap': 0.59975, 'Black-box': 0.6752,  'Boundary dist': 0.665, 'SIF': 0.69795},  # 20k
    '7': {'Gap': 0.59072, 'Black-box': 0.65872, 'Boundary dist': 0.667, 'SIF': 0.67836},  # 25k
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
plt.ylim(bottom=0.6, top=0.95)
plt.xticks(index + 2.5*bar_width, ('1', '2', '3', '4', '5', '6', '7'))
# plt.yticks([0.5, 0.6, 0.7, 0.8])
plt.title('CIFAR-10')
plt.legend((rects1, rects2, rects3, rects4), ('Gap', 'Black-box', 'Boundary dist', 'SIF (ours)'),
           loc=(0.63, 0.77), ncol=1, fancybox=True, prop={'size': 10})
plt.tight_layout()
plt.savefig('cifar10_densenet_attack_scores.png', dpi=350)
plt.show()
