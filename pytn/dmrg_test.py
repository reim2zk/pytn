import numpy as np
import tensorflow as tf

import pytn.mps
import pytn.mpo
import pytn.dmrg


def test_run():
    num = 5
    psi0 = tf.ones([2]*num)
    print(psi0.dtype)
    psi0 = tf.constant(
        np.random.uniform(size=2**num).reshape([2]*num),
        dtype=tf.float32)
    print(psi0.dtype)
    mps0 = pytn.mps.canonical_mps(psi0)
    mps0.normalize()
    mpo0 = pytn.mpo.ising_transfer_matrix_mpo(1.0, num)

    t = mpo0.full_contract()
    m = mpo0.as_matrix()
    print("tensor", t.shape)
    print("matrix")
    print(m)

    lam_x, U_ix = tf.linalg.eigh(m)
    print("lambda")
    print(lam_x)

    dmrg0 = pytn.dmrg.DMRG(mpo0, mps0)
    dmrg0.run()
