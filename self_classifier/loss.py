import tensorflow as tf 

from tensorflow.keras.activations import softmax

from tensorflow.math import log as tf_log
from tensorflow.math import reduce_sum as tf_sum
from tensorflow.math import maximum as tf_max


class SelfClassifier():
    def __init__(self, n_batch: int, n_class: int, t_r: float, t_c: float, eps=1e-8):
        self.N = n_batch
        self.C = n_class
        self.t_r = t_r
        self.t_c = t_c
        self.eps = eps

    @tf.function
    def __call__(self, s1, s2):
        N, C, t_r, t_c = self.N, self.C, self.t_r, self.t_c
        # log_y_x1
        softmax_y_x1 = softmax(s1/t_r, axis=1)

        norm_y_x1 = self._normalize(softmax_y_x1, p=1, dim=0)
        log_y_x1 = tf_log(N/C * norm_y_x1 + self.eps)

        # log_y_x2
        softmax_y_x2 = softmax(s2/t_r, axis=1)
        norm_y_x2 = self._normalize(softmax_y_x2, p=1, dim=0)
        log_y_x2 = tf_log(N/C * norm_y_x2 + self.eps)

        y_x1 = self._normalize(softmax(s1/t_c, axis=0), p=1, dim=1)
        y_x2 = self._normalize(softmax(s2/t_c, axis=0), p=1, dim=1)

        l1 = - tf_sum(y_x2*log_y_x1, axis=-1) / N
        l2 = - tf_sum(y_x1 * log_y_x2, axis=-1) / N
        l1_l2 = (l1 + l2) / 2
        L = tf_sum(l1_l2)
        return L

    def _normalize(self, tensor, p=2, dim=1):

        norm = tf.norm(tensor, ord=p, axis=dim)
        t_eps = tf.constant([self.eps])
        if dim == 0:
            norm_tensor = tensor / tf_max(norm, t_eps)
        elif dim == 1:
            norm_tensor = tensor / tf.reshape(tf_max(norm, t_eps), shape=[-1,1])
            
        return norm_tensor
        