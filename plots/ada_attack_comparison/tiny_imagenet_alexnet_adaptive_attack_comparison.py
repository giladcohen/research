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
    '2': {'Gap': 0.982,   'Black-box': 0.758,   'Boundary dist': 0.983, 'SIF': 0.98},   # 1k
    '3': {'Gap': 0.96,    'Black-box': 0.9408,  'Boundary dist': 0.97,  'SIF': 0.9804},  # 5k
    '4': {'Gap': 0.5679,  'Black-box': 0.5107,  'Boundary dist': 0.554, 'SIF': 0.5679},  # 10k
    '5': {'Gap': 0.565066666666666,  'Black-box': 0.503533333333333,  'Boundary dist': 0.561, 'SIF': 0.564666666666666},  # 15k
    '6': {'Gap': 0.9343,  'Black-box': 0.90485, 'Boundary dist': 0.956, 'SIF': 0.9476},  # 20k
    '7': {'Gap': 0.91664, 'Black-box': 0.8922,  'Boundary dist': 0.916, 'SIF': 0.93292},  # 25k
}}

attack_score_adaptive_dict = {dataset: {
    '2': {'Gap': 0.982,   'Black-box': 0.758,   'Boundary dist': 0.983, 'SIF': 0.985},   # 1k
    '3': {'Gap': 0.96,    'Black-box': 0.9408,  'Boundary dist': 0.97,  'SIF': 0.9938},  # 5k
    '4': {'Gap': 0.5679,  'Black-box': 0.5107,  'Boundary dist': 0.554, 'SIF': 0.5622},  # 10k
    '5': {'Gap': 0.565066666666666,  'Black-box': 0.503533333333333,  'Boundary dist': 0.561, 'SIF': 0.5674},  # 15k
    '6': {'Gap': 0.9343,  'Black-box': 0.90485, 'Boundary dist': 0.956, 'SIF': 0.9856},  # 20k
    '7': {'Gap': 0.91664, 'Black-box': 0.8922,  'Boundary dist': 0.916, 'SIF': 0.9716},  # 25k
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
plt.ylim(bottom=0.47, top=1.01)
plt.xticks(index + 2.5*bar_width, ('2', '3', '4', '5', '6', '7'))
# plt.yticks([0.8, 0.9, 1.0])
plt.title('Tiny ImageNet')
# plt.legend((rects1, rects2, rects3, rects4, colorless_patch), ('Gap', 'Black-box', 'Boundary dist', 'SIF (ours)', 'Adaptive'),
#            loc=(0.63, 0.70), ncol=1, fancybox=True, prop={'size': 10})
plt.tight_layout()
plt.savefig('tiny_imagenet_alexnet_adaptive_attack_scores.png', dpi=350)
plt.show()
