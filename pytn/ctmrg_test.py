import pytest

import pytn.ctmrg


def test_ctmrg():
    # self, N: int, J_kT: float, chi: int, boundary_cond="fixed")

    ctmrg = pytn.ctmrg.CTMRG(
        N=10,
        J_kT=1.0/2.0,
        chi=None,
        boundary_cond="fixed",
        rescale=False)
    print("CTMRG")
    # ctmrg.run()
    print("direct")
    ctmrg.run_directory()

    print("direct, rescale")
    ctmrg = pytn.ctmrg.CTMRG(
        N=20,
        J_kT=1.0/2.0,
        chi=None,
        boundary_cond="fixed",
        rescale=True)
    ctmrg.run_directory()
    assert 1 == pytest.approx(0.0)
