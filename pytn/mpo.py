import dataclasses
import numpy as np
import tensorflow as tf

import pytn.utils


def ising_transfer_matrix_mpo(J_kT, num):
    w = pytn.utils.ising_tensor_W(J_kT, num)
    return MPO([w for i in range(num)])


def traverse_field_ising_mpo(J, g, num):
    """ MPO representation of traverse field ising model Hamiltonian
        H = -J sum_j (sigma^z_j sigma^z_{j+1} + g sigma^z_j)
    """
    sig_x = np.array([
        [0.0, 1.0],
        [1.0, 0.0]
    ])
    sig_z = np.array([
        [1.0, 0.0],
        [0.0, -1.0]
    ])
    one = np.eye(2)
    zero = np.zeros(shape=(2, 2))
    w_g_h_a_b = tf.constant([
        [one,     zero,  zero],
        [sig_z,   zero,  zero],
        [g*sig_x, sig_z, one]
    ], dtype=tf.float32)  # shape = (3, 3, 2, 2)
    w_g_a_b_h = tf.transpose(w_g_h_a_b, perm=[0, 2, 3, 1])
    w_a_b_h = -J*w_g_a_b_h[2:3, :, :, :]
    w_g_a_b = w_g_a_b_h[:, :, :, 0:1]
    w_li = [w_a_b_h] + [w_g_a_b_h]*(num-2) + [w_g_a_b]
    return MPO(w_li)


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
