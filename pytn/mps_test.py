import pytest
import numpy as np
import tensorflow as tf

from pytn.mps import up_spin_mps, canonical_mps, svd_with_reduction


psi0 = tf.constant(
    [
        [
            [[1.1, 2.2], [2.0, 3.0]],
            [[1.0, 2.1], [1.0, 1.0]]
        ],
        [
            [[1.6, 2.2], [4.0, 2.3]],
            [[2.9, 1.0], [2.2, 1.0]]
        ]
    ])


def tensor_almost_equal(t0: tf.Tensor, t1: tf.Tensor, abs=0.00001):
    return np.abs((t0-t1).numpy()).max() == pytest.approx(0.0, abs=abs)


def test_svd_with_reduction():
    xi_li = [0.1, 0.2, 0.3]
    xj_li = [0.0, 0.15, 0.33, 0.41]
    psi_i_j = tf.constant(
        [[np.sin(xi+xj) for xi in xi_li] for xj in xj_li],
        dtype=tf.float32)
    (full_lam_w, _, _) = tf.linalg.svd(psi_i_j)
    (lam_w, _, _) = svd_with_reduction(psi_i_j, chi=2)
    assert np.max(lam_w) == pytest.approx(np.max(full_lam_w))


def test_spin_up():
    mps = up_spin_mps(5)
    t = mps.full_contract()
    assert np.sum(t.numpy()) == pytest.approx(1.0)
    assert t[0][0][0][0][0].numpy() == pytest.approx(0.0)
    assert t[1][1][1][1][1].numpy() == pytest.approx(1.0)


def test_normalization():
    mps = up_spin_mps(5)
    mps.normalize()
    assert mps.norm2() == pytest.approx(1.0)


def test_move_left():
    mps = canonical_mps(psi0)
    mps.normalize()
    t0 = mps.full_contract()
    assert len(mps.A_li) == 3
    assert len(mps.B_li) == 1

    mps.move_left()
    t1 = mps.full_contract()
    assert len(mps.A_li) == 2
    assert len(mps.B_li) == 2
    assert tensor_almost_equal(t0, t1)

    mps.move_left()
    assert len(mps.A_li) == 1
    assert len(mps.B_li) == 3
    assert tensor_almost_equal(t0, t1)


def test_move_right():
    mps = canonical_mps(psi0)
    mps.normalize()
    t0 = mps.full_contract()
    mps.move_left()
    mps.move_left()
    mps.move_right()
    mps.move_right()
    t1 = mps.full_contract()
    assert tensor_almost_equal(t0, t1)


def test_move_left_edge():
    mps0 = canonical_mps(psi0)
    mps1 = up_spin_mps(5)
    mps2 = up_spin_mps(8, chi=10)
    for mps in [mps0, mps1, mps2]:
        mps.normalize()
        n = mps.num_A() + mps.num_B()
        t0 = mps.full_contract()

        mps.move_left_edge()
        t1 = mps.full_contract()
        assert 1 == mps.num_A()
        assert n-1 == mps.num_B()
        assert tensor_almost_equal(t0, t1)

        mps.move_right_edge()
        t2 = mps.full_contract()
        assert n-1 == mps.num_A()
        assert 1 == mps.num_B()
        assert tensor_almost_equal(t0, t2)


def test_mps_reduce():
    num = 10
    chi = 30

    psi0 = tf.random.uniform(shape=[2]*num)
    mps = canonical_mps(psi0)
    mps.normalize()
    assert mps.norm2() == pytest.approx(1.0)

    mps.reduce(chi)
    assert mps.norm2() == pytest.approx(1.0)

    # assert 1 == pytest.approx(0.0)
