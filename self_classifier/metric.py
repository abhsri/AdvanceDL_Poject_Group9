import sklearn as skl
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

