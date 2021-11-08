import numpy as np
import tensorflow as tf
import pytest

import pytn.mps
import pytn.mpo
import pytn.dmrg


def test_dmrg_run():
    num = 5
    # psi0 = tf.ones([2]*num)
    psi0 = tf.random.uniform(shape=[2]*num)
    print(psi0.dtype)
    psi0 = tf.constant(
        np.random.uniform(size=2**num).reshape([2]*num),
        dtype=tf.float32)
    print(psi0.dtype)
    mps0 = pytn.mps.canonical_mps(psi0)
    mps0.normalize()

    mpo0 = pytn.mpo.traverse_field_ising_mpo(1.0, 1.0, num)

    m = mpo0.as_matrix()

    lam_x, _ = tf.linalg.eigh(m)
    print("exact eigenvalues")
    print(lam_x)

    dmrg0 = pytn.dmrg.DMRG(mpo0, mps0)
    dmrg0.run()

    assert 1 == pytest.approx(2.0)
