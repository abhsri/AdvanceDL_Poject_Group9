from tensorflow.keras.activations import softmax

from tensorflow.math import log as tf_log
from tensorflow.math import reduce_sum as tf_sum
from tensorflow.math import scalar_mul as tf_smul

from tensorflow.linalg import normalize as tf_norm


class SelfClassifier():
    def __init__(self, n_batch: int, n_class: int, t_r: float, t_c: float):
        self.N = n_batch
        self.C = n_class
        self.t_r = t_r
        self.t_c = t_c

    def __call__(self, s1, s2):
        N, C, t_r, t_c = self.N, self.C, self.t_r, self.t_c
        # log_y_x1
        softmax_y_x1 = softmax(s1/t_r, axis=1)
        norm_y_x1, _ = tf_norm(softmax_y_x1, ord=1, axis=0)
        log_y_x1 = tf_log(N/C * norm_y_x1)

        # log_y_x2
        softmax_y_x2 = softmax(s2/t_r, axis=1)
        norm_y_x2, _ = tf_norm(softmax_y_x2, ord=1, axis=0)
        log_y_x2 = tf_log(N/C * norm_y_x2)

        y_x1, _ = tf_norm(softmax(s1/t_c, axis=0), ord=1, axis=1)
        y_x2, _ = tf_norm(softmax(s2/t_c, axis=0), ord=1, axis=1)

        l1 = - tf_sum(y_x2*log_y_x1, axis=-1) / N
        l2 = - tf_sum(y_x1 * log_y_x2, axis=-1) / N
        L = tf_sum((l1 + l2) / 2)
        return L
