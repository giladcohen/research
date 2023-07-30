import numpy as np
import os

from research.datasets.utils import get_robustness_inds

rand_gen = np.random.RandomState(123456)
dataset_dirs = [
    '/Users/giladcohen/data/gilad/logs/glove_emb/cifar10',
    '/Users/giladcohen/data/gilad/logs/glove_emb/cifar100',
    '/Users/giladcohen/data/gilad/logs/glove_emb/svhn',
    '/Users/giladcohen/data/gilad/logs/glove_emb/tiny_imagenet'
]

def generate_robustness_inds():
    for dataset_dir in dataset_dirs:
        os.makedirs(dataset_dir, exist_ok=True)
        if 'svhn' in dataset_dir:
            test_test_size, test_size = 10000, 26000
        else:
            test_test_size, test_size = 10000, 10000

        test_test_inds = rand_gen.choice(np.arange(test_size), test_test_size, replace=False)
        test_test_inds.sort()
        test_val_inds = np.asarray([i for i in np.arange(test_size) if i not in test_test_inds])

        np.save(os.path.join(dataset_dir, 'test_test_inds.npy'), test_test_inds)
        np.save(os.path.join(dataset_dir, 'test_val_inds.npy'), test_val_inds)

def generate_detection_inds():
    for dataset_dir in dataset_dirs:
        dataset = os.path.split(dataset_dir)[-1]
        _, test_test_inds = get_robustness_inds(dataset)
        val_size = int(0.8 * len(test_test_inds))

        detection_val_inds = rand_gen.choice(test_test_inds, val_size, replace=False)
        detection_val_inds.sort()
        detection_test_inds = np.asarray([i for i in test_test_inds if i not in detection_val_inds])

        np.save(os.path.join(dataset_dir, 'detection_test_inds.npy'), detection_test_inds)
        np.save(os.path.join(dataset_dir, 'detection_val_inds.npy'), detection_val_inds)


generate_robustness_inds()
generate_detection_inds()
print('done')
