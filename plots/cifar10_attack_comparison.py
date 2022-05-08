"""Plotting the MI attack results for CIFAR10"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

datasets = ['cifar10', 'cifar100', 'svhn']
target_models = ['1', '2', '3', '4', '5', '6', '7']
attacks  = ['Gap', 'Black-box', 'Boundary dist', 'SIF']

dataset = datasets[0]
n_groups = len(target_models)

attack_score_dict = {dataset: {
    '1': {'Gap': 0.89,    'Black-box': 0.54,    'Boundary dist': 0.95,  'SIF': 0.99},    # 100
    '2': {'Gap': 0.807,   'Black-box': 0.808,   'Boundary dist': 0.904, 'SIF': 0.951},   # 1k
    '3': {'Gap': 0.7068,  'Black-box': 0.833,   'Boundary dist': 0.849, 'SIF': 0.9088},  # 5k
    '4': {'Gap': 0.678,   'Black-box': 0.8159,  'Boundary dist': 0.799, 'SIF': 0.8689},  # 10k
    '5': {'Gap': 0.627,   'Black-box': 0.7572,  'Boundary dist': 0.769, 'SIF': 0.813133333333333},  # 15k
    '6': {'Gap': 0.62205, 'Black-box': 0.7583,  'Boundary dist': 0.737, 'SIF': 0.8001},  # 20k
    '7': {'Gap': 0.61556, 'Black-box': 0.78896, 'Boundary dist': 0.764, 'SIF': 0.7766},  # 25k
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
plt.ylim(bottom=0.46, top=1.04)
plt.xticks(index + 2.5*bar_width, ('1', '2', '3', '4', '5', '6', '7'))
plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.title('CIFAR-10')
plt.legend((rects1, rects2, rects3, rects4), ('Gap', 'Black-box', 'Boundary dist', 'SIF (ours)'),
           loc=(0.63, 0.77), ncol=1, fancybox=True, prop={'size': 10})
plt.tight_layout()
plt.show()
# plt.savefig('cifar10_attack_scores.png', dpi=350)

