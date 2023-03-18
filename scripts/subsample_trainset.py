import numpy as np
import os

def randomize_max_p_points(x: np.ndarray, y: np.ndarray, p: int):
    if x.shape[0] > p:
        inds = np.random.choice(x.shape[0], p, replace=False)
        return x[inds], y[inds]
    else:
        return x, y

DIR = '/data/gilad/logs/mi/cifar10/resnet18/relu/s_25k_wo_aug/data'
OUT_DIR = os.path.join(os.path.dirname(DIR), 'data_train_2500_test_2500')

X_member_train = np.load(os.path.join(DIR, 'X_member_train.npy'))
y_member_train = np.load(os.path.join(DIR, 'y_member_train.npy'))
X_non_member_train = np.load(os.path.join(DIR, 'X_non_member_train.npy'))
y_non_member_train = np.load(os.path.join(DIR, 'y_non_member_train.npy'))

X_member_train, y_member_train = randomize_max_p_points(X_member_train, y_member_train, 2500)
X_non_member_train, y_non_member_train = randomize_max_p_points(X_non_member_train, y_non_member_train, 2500)

np.save(os.path.join(OUT_DIR, 'X_member_train.npy'), X_member_train)
np.save(os.path.join(OUT_DIR, 'y_member_train.npy'), y_member_train)
np.save(os.path.join(OUT_DIR, 'X_non_member_train.npy'), X_non_member_train)
np.save(os.path.join(OUT_DIR, 'y_non_member_train.npy'), y_non_member_train)

print('done')
