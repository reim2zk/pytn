import pytest

import pytn.ctmrg


def test_ctmrg():
    # self, N: int, J_kT: float, chi: int, boundary_cond="fixed")

    # ctmrg.run()
    print("direct")
    ctmrg = pytn.ctmrg.CTMRG(
        N=10,
        J_kT=1.0/2.0,
        chi=None,
        boundary_cond="fixed",
        rescale=False)

    res = ctmrg.run_directory()
    for el in res:
        print(el)

    print("direct, rescale")
    ctmrg = pytn.ctmrg.CTMRG(
        N=20,
        J_kT=1.0/2.0,
        chi=None,
        boundary_cond="fixed",
        rescale=True)
    res = ctmrg.run_directory()
    for el in res:
        print(el)

    print("CTMRG(chi=10)")
    ctmrg = pytn.ctmrg.CTMRG(
        N=20,
        J_kT=1.0/2.0,
        chi=10,
        boundary_cond="fixed",
        rescale=True,
        disp=False)
    res = ctmrg.run()
    for el in res:
        print(el)
    assert 1 == pytest.approx(0.0)
