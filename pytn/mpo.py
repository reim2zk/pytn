import dataclasses
import numpy as np
import tensorflow as tf

import pytn.utils


def ising_transfer_matrix_mpo(J_kT, num):
    w = pytn.utils.ising_tensor_W(J_kT, num)
    return MPO([w for i in range(num)])


@dataclasses.dataclass
class MPO:
    W_li: tf.Tensor  # list of tensors W(g)^m_a(h)

    def full_contract(self):
        X_x_m_a_y = None
        m_li = []
        a_li = []
        for W_y_n_b_z in self.W_li:
            (_, n, b, _) = W_y_n_b_z.shape
            m_li.append(n)
            a_li.append(b)
            if X_x_m_a_y is None:
                X_x_m_a_y = W_y_n_b_z
            else:
                X_x_m_n_a_b_y = tf.einsum(
                    "xmay,ynbz->xmnabz", X_x_m_a_y, W_y_n_b_z)
                (x, m, n, a, b, y) = X_x_m_n_a_b_y.shape
                X_x_m_a_y = tf.reshape(X_x_m_n_a_b_y,
                                       shape=[x, m*n, a*b, y])
        X_m_a = tf.einsum("xmay->ma", X_x_m_a_y)
        X = tf.reshape(X_m_a, shape=m_li+a_li)
        return X

    def as_matrix(self):
        t = self.full_contract()
        n_li = list(t.shape)
        num = len(n_li)//2
        n = np.product(n_li[:num])
        print("n=", n)
        return tf.reshape(t, shape=(n, n))
