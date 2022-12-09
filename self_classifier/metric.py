import sklearn as skl
import matplotlib.pyplot as plt
import scipy
import numpy as np


def metric(y_true, y_pred, is_print=False):
    nmi = skl.metrics.normalized_mutual_info_score(y_true, y_pred)
    ami = skl.metrics.adjusted_mutual_info_score(y_true, y_pred)
    ari = skl.metrics.adjusted_rand_score(y_true, y_pred)
    acc = None

    # there might be a problem when there are more class than labels
    confusion_matrix = skl.metrics.confusion_matrix(y_true, y_pred)
    row, col = scipy.optimize.linear_sum_assignment(
        confusion_matrix, maximize=True)
    ln = len(confusion_matrix)
    p = np.zeros((ln, ln))
    for r, c in zip(row, col):
        p[r, c] = 1

    new_confusion_matrix = np.linalg.inv(p)@confusion_matrix

    acc = np.trace(new_confusion_matrix) / len(y_pred)
    return nmi, ami, ari, acc


def class_histogram(labels, plot=True, addon_title=''):
    cls_list, cls_hist = np.unique(labels, return_counts=True)
    cls_cnt = cls_list.shape[0]

    if plot:
        fig, axs = plt.subplots(1, figsize=(13, 6))
        axs = plt.hist(labels, bins=cls_cnt)
        plt.title(f'Class Histogram: {cls_cnt} for {addon_title}')
        plt.xlabel('Class')
        plt.ylabel('Datapoints in set')
        plt.show()
    return cls_cnt, cls_list, cls_hist