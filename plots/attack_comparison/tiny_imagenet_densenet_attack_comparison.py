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
    '2': {'Gap': 0.985,   'Black-box': 0.856,   'Boundary dist': 0.986, 'SIF': 0.987},   # 1k
    '3': {'Gap': 0.6876,  'Black-box': 0.5454,  'Boundary dist': 0.675, 'SIF': 0.6866},  # 5k
    '4': {'Gap': 0.6061,  'Black-box': 0.5094,  'Boundary dist': 0.612, 'SIF': 0.6057},  # 10k
    '5': {'Gap': 0.561733333333333,  'Black-box': 0.500066666666666,  'Boundary dist': 0.549, 'SIF': 0.561533333333333},  # 15k
    '6': {'Gap': 0.63965, 'Black-box': 0.5186,  'Boundary dist': 0.625, 'SIF': 0.63935},  # 20k
    '7': {'Gap': 0.6046,  'Black-box': 0.50644, 'Boundary dist': 0.57,  'SIF': 0.60472},  # 25k
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
plt.ylim(bottom=0.49, top=1.002)
plt.xticks(index + 2.5*bar_width, ('2', '3', '4', '5', '6', '7'))
# plt.yticks([0.9, 0.95, 1.0])
# plt.legend((rects1, rects2, rects3, rects4), ('Gap', 'Black-box', 'Boundary dist', 'SIF (ours)'),
#            loc=(0.63, 0.77), ncol=1, fancybox=True, prop={'size': 10})
plt.title('Tiny ImageNet')
plt.tight_layout()
plt.savefig('tiny_imagenet_densenet_attack_scores.png', dpi=350)
plt.show()

