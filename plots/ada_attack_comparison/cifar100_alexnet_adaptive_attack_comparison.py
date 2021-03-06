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
    '1': {'Gap': 0.8,     'Black-box': 0.43,    'Boundary dist': 0.8,   'SIF': 0.79},   # 100
    '2': {'Gap': 0.961,   'Black-box': 0.8,     'Boundary dist': 0.959, 'SIF': 0.963},   # 1k
    '3': {'Gap': 0.7176,  'Black-box': 0.5492,  'Boundary dist': 0.733, 'SIF': 0.717},  # 5k
    '4': {'Gap': 0.7142,  'Black-box': 0.5586,  'Boundary dist': 0.712, 'SIF': 0.7142},  # 10k
    '5': {'Gap': 0.8606,  'Black-box': 0.850933333333333,  'Boundary dist': 0.888, 'SIF': 0.8924},  # 15k
    '6': {'Gap': 0.8422,  'Black-box': 0.8415,  'Boundary dist': 0.886, 'SIF': 0.86515},  # 20k
    '7': {'Gap': 0.81636, 'Black-box': 0.84428, 'Boundary dist': 0.859, 'SIF': 0.84644},  # 25k
}}

attack_score_adaptive_dict = {dataset: {
    '1': {'Gap': 0.8,     'Black-box': 0.43,    'Boundary dist': 0.8,   'SIF': 0.79},   # 100
    '2': {'Gap': 0.961,   'Black-box': 0.8,     'Boundary dist': 0.959, 'SIF': 0.965},   # 1k
    '3': {'Gap': 0.7176,  'Black-box': 0.5492,  'Boundary dist': 0.733, 'SIF': 0.7168},  # 5k
    '4': {'Gap': 0.7142,  'Black-box': 0.5586,  'Boundary dist': 0.712, 'SIF': 0.7148},  # 10k
    '5': {'Gap': 0.8606,  'Black-box': 0.850933333333333,  'Boundary dist': 0.888, 'SIF': 0.95},  # 15k
    '6': {'Gap': 0.8422,  'Black-box': 0.8415,  'Boundary dist': 0.886, 'SIF': 0.9122},  # 20k
    '7': {'Gap': 0.81636, 'Black-box': 0.84428, 'Boundary dist': 0.859, 'SIF': 0.9238},  # 25k
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
values1 = [attack_score_dict[dataset]['1']['Gap'], attack_score_dict[dataset]['2']['Gap'],
           attack_score_dict[dataset]['3']['Gap'], attack_score_dict[dataset]['4']['Gap'],
           attack_score_dict[dataset]['5']['Gap'], attack_score_dict[dataset]['6']['Gap'],
           attack_score_dict[dataset]['7']['Gap']]
rects1 = plt.bar(index + bar_width, values1, bar_width,
                 alpha=opacity,
                 color='black',
                 edgecolor='black',
                 label='Gap')

# attack rectangle: all of Black-box
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

# attack rectangle: all of Boundary distance
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

# attack rectangle: all of SIF
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

values44 = [adaptive_boost_dict[dataset]['1']['SIF'], adaptive_boost_dict[dataset]['2']['SIF'],
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
plt.ylim(bottom=0.4, top=1.0)
plt.xticks(index + 2.5*bar_width, ('1', '2', '3', '4', '5', '6', '7'))
# plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.title('CIFAR-100')
# plt.legend((rects1, rects2, rects3, rects4, colorless_patch), ('Gap', 'Black-box', 'Boundary dist', 'SIF (ours)', 'Adaptive'),
#            loc=(0.63, 0.70), ncol=1, fancybox=True, prop={'size': 10})
plt.tight_layout()
plt.savefig('cifar100_alexnet_adaptive_attack_scores.png', dpi=350)
plt.show()
