import pytest

import pytn.mpo


def test_traverse_field_ising_mpo():
    mpo = pytn.mpo.traverse_field_ising_mpo(1.0, 4)
    print([w.shape for w in mpo.W_li])
    assert 3 == pytest.approx(1.0)
