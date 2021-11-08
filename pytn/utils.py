import tensorflow as tf


def ising_tensor_W(J_kT: float, num: int):
    #   W_gmah
    #     m
    # g <-+-> h
    #     a
    n = tf.newaxis
    s = tf.constant([1.0, -1.0])
    s = (
        s[:, n, n, n] * s[n, :, n, n] +
        s[n, :, n, n] * s[n, n, n, :] +
        s[n, n, :, n] * s[n, n, n, :] +
        s[:, n, n, n] * s[n, n, :, n]
    )
    w = tf.exp(J_kT * s)
    return w
