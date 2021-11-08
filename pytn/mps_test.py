import pytest
import numpy as np
import tensorflow as tf

from pytn.mps import up_spin_mps, canonical_mps


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


def test_spin_up():
    mps = up_spin_mps(5)
    t = mps.full_contract()
    print(t.shape)
    print(t[0][0][0][0][0].numpy())
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
    mps = canonical_mps(psi0)
    mps.normalize()
    t0 = mps.full_contract()

    mps.move_left_edge()
    t1 = mps.full_contract()
    assert 1 == mps.num_A()
    assert 3 == mps.num_B()
    assert tensor_almost_equal(t0, t1)

    mps.move_right_edge()
    t2 = mps.full_contract()
    assert 3 == mps.num_A()
    assert 1 == mps.num_B()
    assert tensor_almost_equal(t0, t2)