import tensorflow as tf
import pytest

import pytn.mps
import pytn.mpo
import pytn.dmrg


def test_dmrg_run():
    num = 8
    chi = 10  # max=32
    # psi0 = tf.constant(
    #     np.random.uniform(size=2**num).reshape([2]*num),
    #     dtype=tf.float32)
    # print(psi0.dtype)
    # mps0 = pytn.mps.canonical_mps(psi0, chi)
    mps0 = pytn.mps.up_spin_mps(num, chi)
    mps0.normalize()
    print(mps0.chi, [a.shape for a in mps0.A_li])

    mpo0 = pytn.mpo.traverse_field_ising_mpo(1.0, 0.1, num)

    m = mpo0.as_matrix()

    lam_x, _ = tf.linalg.eigh(m)
    print("exact eigenvalues")
    print(lam_x)

    dmrg0 = pytn.dmrg.DMRG(mpo0, mps0, 10)
    dmrg0.run()

    assert 1 == pytest.approx(2.0)
