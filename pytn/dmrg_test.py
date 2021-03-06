import numpy as np
import tensorflow as tf
import pytest

import pytn.mps
import pytn.mpo
import pytn.dmrg


def tensor_almost_equal(t0: tf.Tensor, t1: tf.Tensor, abs=0.00001):
    return np.abs((t0-t1).numpy()).max() == pytest.approx(0.0, abs=abs)


def test_dmrg_env_matrix():
    num = 8
    chi = 10  # max=32

    mps = pytn.mps.up_spin_mps(num, chi)

    mpo = pytn.mpo.traverse_field_ising_mpo(1.0, 0.1, num)
    dmrg = pytn.dmrg.DMRG(mpo, mps)

    dmrg.init()
    for i in [0, 1]:
        for _ in range(i):
            dmrg.step_right()
        Y_xayb = dmrg.env_matrix_moving_right()
        [x, a, y, b] = Y_xayb.shape[:4]
        n = x*a*y*b
        Y_mat = tf.reshape(Y_xayb, shape=(n, n))
        assert tensor_almost_equal(Y_mat, tf.transpose(Y_mat))

    assert 1.0 == pytest.approx(2.0)


def test_dmrg_run():
    num = 8
    for chi in [4]:
        mps0 = pytn.mps.up_spin_mps(num, chi)
        print(mps0.chi, [a.shape for a in mps0.A_li])

        mpo0 = pytn.mpo.traverse_field_ising_mpo(1.0, 0.5, num)

        m = mpo0.as_matrix()
        lam_x, _ = tf.linalg.eigh(m)
        print("exact eigenvalues:")
        print(lam_x[0:5])
        dmrg0 = pytn.dmrg.DMRG(mpo0, mps0)
        res_li = dmrg0.run()

        print("calculated", res_li[-1]["min_eigval"])
        assert res_li[-1]["min_eigval"] == pytest.approx(lam_x[0], 0.00001)
