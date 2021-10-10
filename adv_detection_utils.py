import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

def train_lr(X, y):
    """
    :param X: the data samples
    :param y: the labels
    :return:
    """
    lr = LogisticRegressionCV(n_jobs=-1, max_iter=20000, cv=3).fit(X, y)
    return lr

def block_split(X, Y):
    """
    Split the data into 80% for training and 20% for testing
    in a block size of 100.
    :param X:
    :param Y:
    :return:
    """
    print("Isolated split 80%, 20% for training and testing")
    num_samples = X.shape[0]
    partition = int(num_samples / 3)
    X_adv, Y_adv = X[:partition], Y[:partition]
    X_norm, Y_norm = X[partition: 2*partition], Y[partition: 2*partition]
    X_noisy, Y_noisy = X[2*partition:], Y[2*partition:]
    num_train = int(partition*0.008) * 100

    X_train = np.concatenate((X_norm[:num_train], X_noisy[:num_train], X_adv[:num_train]))
    Y_train = np.concatenate((Y_norm[:num_train], Y_noisy[:num_train], Y_adv[:num_train]))

    X_test = np.concatenate((X_norm[num_train:], X_noisy[num_train:], X_adv[num_train:]))
    Y_test = np.concatenate((Y_norm[num_train:], Y_noisy[num_train:], Y_adv[num_train:]))

    return X_train, Y_train, X_test, Y_test

def random_split(X, Y):
    """
    Random split the data into 80% for training and 20% for testing
    :param X:
    :param Y:
    :return:
    """
    print("random split 80%, 20% for training and testing")
    num_samples = X.shape[0]
    num_train = int(num_samples * 0.8)
    rand_pert = np.random.permutation(num_samples)
    X = X[rand_pert]
    Y = Y[rand_pert]
    X_train, X_test = X[:num_train], X[num_train:]
    Y_train, Y_test = Y[:num_train], Y[num_train:]

    return X_train, Y_train, X_test, Y_test

def compute_roc(y_true, y_pred, plot=False):
    """
    :param y_true: ground truth
    :param y_pred: predictions
    :param plot:
    :return:
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score
