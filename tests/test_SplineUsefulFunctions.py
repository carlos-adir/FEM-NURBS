import pytest
# import unittest
import numpy as np
import femnurbs.SplineUsefulFunctions as SUF


def test_isValidU():
    with pytest.raises(TypeError):
        SUF.isValidU()

    assert SUF.isValidU(0) is False
    assert SUF.isValidU(1.2) is False
    assert SUF.isValidU({}) is False
    assert SUF.isValidU(-1) is False
    assert SUF.isValidU({1: 1}) is False
    assert SUF.isValidU([0, 0, 0, 1, 1]) is False
    assert SUF.isValidU([0, 0, 1, 1, 1, ]) is False
    assert SUF.isValidU([0, 0, 0, 0, 1, 1, 1]) is False
    assert SUF.isValidU([0, 0, 0, 1, 1, 1, 1]) is False
    assert SUF.isValidU([-1, -1, 1, 1]) is False
    assert SUF.isValidU([0, 0, 2, 2]) is False

    assert SUF.isValidU([0, 0, 0.8, 0.2, 1, 1]) is False
    assert SUF.isValidU([0, 0, 0, 1, 0.5, 1, 1]) is False

    assert SUF.isValidU([0, 0, 1, 1]) is True
    assert SUF.isValidU([0, 0, 0, 1, 1, 1]) is True
    assert SUF.isValidU([0, 0, 0, 0, 1, 1, 1, 1]) is True
    assert SUF.isValidU([0, 0, 0.2, 0.8, 1, 1]) is True
    assert SUF.isValidU([0, 0, 0, 0.5, 1, 1, 1]) is True
    assert SUF.isValidU([0, 0, 0.1, 0.5, 0.9, 1, 1]) is True
    assert SUF.isValidU([0, 0, 0.5, 0.5, 1, 1]) is True

    assert SUF.isValidU([0, 0, 0.5, 0.5, 0.5, 1, 1]) is False

def test_UBezier():

    for p in range(1, 10):
        assert SUF.isValidU(SUF.UBezier(p=p)) is True

    Ugood = np.array([0, 0, 1, 1])
    Utest = SUF.UBezier(p=1)
    np.testing.assert_allclose(Ugood, Utest)

    Ugood = np.array([0, 0, 0, 1, 1, 1])
    Utest = SUF.UBezier(p=2)
    np.testing.assert_allclose(Ugood, Utest)

    Ugood = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    Utest = SUF.UBezier(p=3)
    np.testing.assert_allclose(Ugood, Utest)


def test_UUniform():
    for p in range(1, 10):
        for n in range(p + 1, 11):
            assert SUF.isValidU(SUF.UUniform(p=p, n=n)) is True

    Ugood = np.array([0, 0, 1, 1])
    Utest = SUF.UUniform(p=1, n=2)
    np.testing.assert_allclose(Ugood, Utest)

    Ugood = np.array([0, 0, 0.5, 1, 1])
    Utest = SUF.UUniform(p=1, n=3)
    np.testing.assert_allclose(Ugood, Utest)

    Ugood = np.array([0, 0, 0.25, 0.5, 0.75, 1, 1])
    Utest = SUF.UUniform(p=1, n=5)
    np.testing.assert_allclose(Ugood, Utest)

    Ugood = np.array([0, 0, 0.2, 0.4, 0.6, 0.8, 1, 1])
    Utest = SUF.UUniform(p=1, n=6)
    np.testing.assert_allclose(Ugood, Utest)

    Ugood = np.array([0, 0, 0, 1, 1, 1])
    Utest = SUF.UUniform(p=2, n=3)
    np.testing.assert_allclose(Ugood, Utest)

    Ugood = np.array([0, 0, 0, 0.5, 1, 1, 1])
    Utest = SUF.UUniform(p=2, n=4)
    np.testing.assert_allclose(Ugood, Utest)

    Ugood = np.array([0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1])
    Utest = SUF.UUniform(p=2, n=6)
    np.testing.assert_allclose(Ugood, Utest)

    Ugood = np.array([0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1])
    Utest = SUF.UUniform(p=2, n=7)
    np.testing.assert_allclose(Ugood, Utest)

    Ugood = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    Utest = SUF.UUniform(p=3, n=4)
    np.testing.assert_allclose(Ugood, Utest)

    Ugood = np.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1])
    Utest = SUF.UUniform(p=3, n=5)
    np.testing.assert_allclose(Ugood, Utest)

    Ugood = np.array([0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1])
    Utest = SUF.UUniform(p=3, n=7)
    np.testing.assert_allclose(Ugood, Utest)

    Ugood = np.array([0, 0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1, 1])
    Utest = SUF.UUniform(p=3, n=8)
    np.testing.assert_allclose(Ugood, Utest)


def test_URandom():
    Ntest = 100
    for p in (1, 2, 3):
        for n in range(p + 1, 30):
            for zz in range(Ntest):
                U = SUF.URandom(p=p, n=n)
                assert SUF.isValidU(U) is True
                assert SUF.getPfromU(U) == p
                assert SUF.getNfromU(U) == n


def test_transpose():
    II = np.eye(3)
    IItest = SUF.transpose(II)
    np.testing.assert_allclose(IItest, II)

    II = np.eye(4)
    IItest = SUF.transpose(II)
    np.testing.assert_allclose(IItest, II)

    II = np.eye(3)
    IItest = SUF.transpose(II, diagonal=2)
    np.testing.assert_allclose(IItest, II)

    II = np.eye(4)
    IItest = SUF.transpose(II, diagonal=2)
    np.testing.assert_allclose(IItest, II)


def test_isSymetric():
    II = np.eye(3)
    assert SUF.isSymetric(II) is True

    II = np.eye(4)
    assert SUF.isSymetric(II) is True

    II = np.eye(3)
    assert SUF.isSymetric(II, diagonal=2) is True

    II = np.eye(4)
    assert SUF.isSymetric(II, diagonal=2) is True

    II = np.array([[1, 2, 3, 4],
                   [4, 3, 2, 1]])
    assert SUF.isSymetric(II, diagonal=2) is True

    II = np.array([[1, 2, 4, 4],
                   [4, 4, 2, 1]])
    assert SUF.isSymetric(II, diagonal=2) is True

    II = np.array([[7, 2, 4, 3],
                   [4, 4, 2, 7]])
    assert SUF.isSymetric(II, diagonal=2) is False

    II = np.array([[7, 2, 4, 7],
                   [7, 4, 2, 3]])
    assert SUF.isSymetric(II, diagonal=2) is False


def test_getPfromU():
    U = SUF.UBezier(p=1)
    ptest = SUF.getPfromU(U)
    assert ptest == 1

    U = SUF.UBezier(p=2)
    ptest = SUF.getPfromU(U)
    assert ptest == 2

    U = SUF.UBezier(p=3)
    ptest = SUF.getPfromU(U)
    assert ptest == 3

    U = SUF.UBezier(p=4)
    ptest = SUF.getPfromU(U)
    assert ptest == 4

    U = SUF.UUniform(p=1, n=6)
    ptest = SUF.getPfromU(U)
    assert ptest == 1

    U = SUF.UUniform(p=2, n=6)
    ptest = SUF.getPfromU(U)
    assert ptest == 2

    U = SUF.UUniform(p=3, n=6)
    ptest = SUF.getPfromU(U)
    assert ptest == 3

    U = SUF.UUniform(p=4, n=6)
    ptest = SUF.getPfromU(U)
    assert ptest == 4

    U = np.array([0, 0, 0, 0.2, 0.8, 1, 1, 1])
    ptest = SUF.getPfromU(U)
    assert ptest == 2


def test_getNfromU():
    U = SUF.UBezier(p=1)
    ptest = SUF.getNfromU(U)
    assert ptest == 2

    U = SUF.UBezier(p=2)
    ptest = SUF.getNfromU(U)
    assert ptest == 3

    U = SUF.UBezier(p=3)
    ptest = SUF.getNfromU(U)
    assert ptest == 4

    U = SUF.UBezier(p=4)
    ptest = SUF.getNfromU(U)
    assert ptest == 5

    U = SUF.UUniform(p=1, n=6)
    ptest = SUF.getNfromU(U)
    assert ptest == 6

    U = SUF.UUniform(p=2, n=6)
    ptest = SUF.getNfromU(U)
    assert ptest == 6

    U = SUF.UUniform(p=3, n=6)
    ptest = SUF.getNfromU(U)
    assert ptest == 6

    U = SUF.UUniform(p=4, n=6)
    ptest = SUF.getNfromU(U)
    assert ptest == 6

    U = np.array([0, 0, 0, 0.2, 0.8, 1, 1, 1])
    ptest = SUF.getNfromU(U)
    assert ptest == 5


def test_transformUtoH():
    U = SUF.UBezier(p=1)
    Hgood = np.array([1])
    Htest = SUF.transformUtoH(U)
    np.testing.assert_allclose(Hgood, Htest)

    U = SUF.UBezier(p=2)
    Hgood = np.array([0, 1, 0])
    Htest = SUF.transformUtoH(U)
    np.testing.assert_allclose(Hgood, Htest)

    U = SUF.UBezier(p=3)
    Hgood = np.array([0, 0, 1, 0, 0])
    Htest = SUF.transformUtoH(U)
    np.testing.assert_allclose(Hgood, Htest)

    U = SUF.UBezier(p=4)
    Hgood = np.array([0, 0, 0, 1, 0, 0, 0])
    Htest = SUF.transformUtoH(U)
    np.testing.assert_allclose(Hgood, Htest)

    U = SUF.UUniform(p=1, n=6)
    Hgood = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    Htest = SUF.transformUtoH(U)
    np.testing.assert_allclose(Hgood, Htest)

    U = SUF.UUniform(p=2, n=6)
    Hgood = np.array([0, 0.25, 0.25, 0.25, 0.25, 0])
    Htest = SUF.transformUtoH(U)
    np.testing.assert_allclose(Hgood, Htest)

    U = SUF.UUniform(p=3, n=6)
    Hgood = np.array([0, 0, 1, 1, 1, 0, 0]) / 3
    Htest = SUF.transformUtoH(U)
    np.testing.assert_allclose(Hgood, Htest)

    U = SUF.UUniform(p=4, n=6)
    Hgood = np.array([0, 0, 0, 1, 1, 0, 0, 0]) / 2
    Htest = SUF.transformUtoH(U)
    np.testing.assert_allclose(Hgood, Htest)

    U = np.array([0, 0, 0, 0.2, 0.8, 1, 1, 1])  # p = 2 and n = 5
    Hgood = np.array([0, 0.2, 0.6, 0.2, 0])
    Htest = SUF.transformUtoH(U)
    np.testing.assert_allclose(Hgood, Htest)

    U = np.array([0, 0, 0, 0, 0.2, 0.8, 1, 1, 1, 1])  # p = 3 and n = 6
    Hgood = np.array([0, 0, 0.2, 0.6, 0.2, 0, 0])
    Htest = SUF.transformUtoH(U)
    np.testing.assert_allclose(Hgood, Htest)

    U = SUF.UBezier(p=2)
    Hgood = np.array([1])
    Htest = SUF.transformUtoH(U, j=0)
    np.testing.assert_allclose(Hgood, Htest)

    U = SUF.UBezier(p=2)
    Hgood = np.array([1])
    Htest = SUF.transformUtoH(U, j=1)
    np.testing.assert_allclose(Hgood, Htest)

    U = SUF.UBezier(p=3)
    Hgood = np.array([1])
    Htest = SUF.transformUtoH(U, j=0)
    np.testing.assert_allclose(Hgood, Htest)

    U = SUF.UBezier(p=3)
    Hgood = np.array([1])
    Htest = SUF.transformUtoH(U, j=1)
    np.testing.assert_allclose(Hgood, Htest)

    U = SUF.UBezier(p=3)
    Hgood = np.array([0, 1, 0])
    Htest = SUF.transformUtoH(U, j=2)
    np.testing.assert_allclose(Hgood, Htest)

    U = SUF.UBezier(p=4)
    Hgood = np.array([1])
    Htest = SUF.transformUtoH(U, j=0)
    np.testing.assert_allclose(Hgood, Htest)

    U = SUF.UBezier(p=4)
    Hgood = np.array([1])
    Htest = SUF.transformUtoH(U, j=1)
    np.testing.assert_allclose(Hgood, Htest)

    U = SUF.UBezier(p=4)
    Hgood = np.array([0, 1, 0])
    Htest = SUF.transformUtoH(U, j=2)
    np.testing.assert_allclose(Hgood, Htest)

    U = SUF.UBezier(p=4)
    Hgood = np.array([0, 0, 1, 0, 0])
    Htest = SUF.transformUtoH(U, j=3)
    np.testing.assert_allclose(Hgood, Htest)

    U = SUF.UUniform(p=1, n=6)
    Hgood = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    Htest = SUF.transformUtoH(U, j=0)
    np.testing.assert_allclose(Hgood, Htest)

    U = SUF.UUniform(p=2, n=6)
    Hgood = np.array([0.25, 0.25, 0.25, 0.25])
    Htest = SUF.transformUtoH(U, j=0)
    np.testing.assert_allclose(Hgood, Htest)

    U = SUF.UUniform(p=2, n=6)
    Hgood = np.array([0.25, 0.25, 0.25, 0.25])
    Htest = SUF.transformUtoH(U, j=1)
    np.testing.assert_allclose(Hgood, Htest)

    U = SUF.UUniform(p=3, n=6)
    Hgood = np.array([1, 1, 1]) / 3
    Htest = SUF.transformUtoH(U, j=0)
    np.testing.assert_allclose(Hgood, Htest)

    U = SUF.UUniform(p=3, n=6)
    Hgood = np.array([1, 1, 1]) / 3
    Htest = SUF.transformUtoH(U, j=1)
    np.testing.assert_allclose(Hgood, Htest)

    U = SUF.UUniform(p=3, n=6)
    Hgood = np.array([0, 1, 1, 1, 0]) / 3
    Htest = SUF.transformUtoH(U, j=2)
    np.testing.assert_allclose(Hgood, Htest)

    U = SUF.UUniform(p=4, n=6)
    Hgood = np.array([1, 1]) / 2
    Htest = SUF.transformUtoH(U, j=0)
    np.testing.assert_allclose(Hgood, Htest)

    U = SUF.UUniform(p=4, n=6)
    Hgood = np.array([1, 1]) / 2
    Htest = SUF.transformUtoH(U, j=1)
    np.testing.assert_allclose(Hgood, Htest)

    U = SUF.UUniform(p=4, n=6)
    Hgood = np.array([0, 1, 1, 0]) / 2
    Htest = SUF.transformUtoH(U, j=2)
    np.testing.assert_allclose(Hgood, Htest)

    U = SUF.UUniform(p=4, n=6)
    Hgood = np.array([0, 0, 1, 1, 0, 0]) / 2
    Htest = SUF.transformUtoH(U, j=3)
    np.testing.assert_allclose(Hgood, Htest)

    U = np.array([0, 0, 0, 0.2, 0.8, 1, 1, 1])  # p = 2 and n = 5
    Hgood = np.array([0.2, 0.6, 0.2])
    Htest = SUF.transformUtoH(U, j=1)
    np.testing.assert_allclose(Hgood, Htest)

    U = np.array([0, 0, 0, 0, 0.2, 0.8, 1, 1, 1, 1])  # p = 3 and n = 6
    Hgood = np.array([0.2, 0.6, 0.2])
    Htest = SUF.transformUtoH(U, j=1)
    np.testing.assert_allclose(Hgood, Htest)

    U = np.array([0, 0, 0, 0, 0.2, 0.8, 1, 1, 1, 1])  # p = 3 and n = 6
    Hgood = np.array([0, 0.2, 0.6, 0.2, 0])
    Htest = SUF.transformUtoH(U, j=2)
    np.testing.assert_allclose(Hgood, Htest)


def test_transformHtoSides():
    H = np.array([1, 1, 1])
    Sgood = np.array([[1], [1]])
    Stest = SUF.transformHtoSides(H)
    np.testing.assert_allclose(Sgood, Stest)

    H = np.array([0, 1, 1])
    Sgood = np.array([[0], [1]])
    Stest = SUF.transformHtoSides(H)
    np.testing.assert_allclose(Sgood, Stest)

    H = np.array([1, 1, 0])
    Sgood = np.array([[1], [0]])
    Stest = SUF.transformHtoSides(H)
    np.testing.assert_allclose(Sgood, Stest)

    H = np.array([0, 1, 0])
    Sgood = np.array([[0], [0]])
    Stest = SUF.transformHtoSides(H)
    np.testing.assert_allclose(Sgood, Stest)

    H = np.array([0.6, 1, 0.3])
    Sgood = np.array([[0.6], [0.3]])
    Stest = SUF.transformHtoSides(H)
    np.testing.assert_allclose(Sgood, Stest)

    H = np.array([6, 10, 3])
    Sgood = np.array([[0.6], [0.3]])
    Stest = SUF.transformHtoSides(H)
    np.testing.assert_allclose(Sgood, Stest)

    H = np.array([1, 1, 1, 1, 1])
    Sgood = np.array([[1, 1], [1, 1]])
    Stest = SUF.transformHtoSides(H)
    np.testing.assert_allclose(Sgood, Stest)

    H = np.array([0, 1, 1, 1, 1])
    Sgood = np.array([[1, 0], [1, 1]])
    Stest = SUF.transformHtoSides(H)
    np.testing.assert_allclose(Sgood, Stest)

    H = np.array([1, 1, 1, 1, 0])
    Sgood = np.array([[1, 1], [1, 0]])
    Stest = SUF.transformHtoSides(H)
    np.testing.assert_allclose(Sgood, Stest)

    H = np.array([0, 0, 1, 0, 0])
    Sgood = np.array([[0, 0], [0, 0]])
    Stest = SUF.transformHtoSides(H)
    np.testing.assert_allclose(Sgood, Stest)

    H = np.array([0.2, 0.6, 1, 0.3, 0.4])
    Sgood = np.array([[0.6, 0.2], [0.3, 0.4]])
    Stest = SUF.transformHtoSides(H)
    np.testing.assert_allclose(Sgood, Stest)

    H = np.array([2, 6, 10, 3, 4])
    Sgood = np.array([[0.6, 0.2], [0.3, 0.4]])
    Stest = SUF.transformHtoSides(H)
    np.testing.assert_allclose(Sgood, Stest)


def test_cutHtoElementZ():
    H = np.array([0.5, 0.5])
    Zgood = np.array([0.5])
    Ztest = SUF.cutHtoElementZ(H, 0)
    np.testing.assert_allclose(Zgood, Ztest)

    H = np.array([0.5, 0.5])
    Zgood = np.array([0.5])
    Ztest = SUF.cutHtoElementZ(H, 1)
    np.testing.assert_allclose(Zgood, Ztest)

    H = np.array([0, 0.5, 0.5, 0])
    Zgood = np.array([0, 0.5, 0.5])
    Ztest = SUF.cutHtoElementZ(H, 0)
    np.testing.assert_allclose(Zgood, Ztest)

    H = np.array([0, 0.5, 0.5, 0])
    Zgood = np.array([0.5, 0.5, 0])
    Ztest = SUF.cutHtoElementZ(H, 1)
    np.testing.assert_allclose(Zgood, Ztest)

    H = np.array([0, 0, 0.5, 0.5, 0, 0])
    Zgood = np.array([0, 0, 0.5, 0.5, 0])
    Ztest = SUF.cutHtoElementZ(H, 0)
    np.testing.assert_allclose(Zgood, Ztest)

    H = np.array([0, 0, 0.5, 0.5, 0, 0])
    Zgood = np.array([0, 0.5, 0.5, 0, 0])
    Ztest = SUF.cutHtoElementZ(H, 1)
    np.testing.assert_allclose(Zgood, Ztest)


def test_isDiagonalDominant():
    M = np.eye(3)
    assert SUF.isDiagonalDominant(M) is True

    M = np.ones((3, 3))
    assert SUF.isDiagonalDominant(M) is False

    M = np.zeros((3, 3))
    assert SUF.isDiagonalDominant(M) is False

    M = np.eye(3) - (1 / 3)
    assert SUF.isDiagonalDominant(M) is False

    M = 1.0001 * np.eye(3) - (1 / 3)
    assert SUF.isDiagonalDominant(M) is True


def test_rafineU():
    if True:
        U = SUF.UBezier(p=1)

        Utest = SUF.rafineU(U, 1)
        Ugood = [0, 1]
        np.testing.assert_allclose(Utest, Ugood)

        Utest = SUF.rafineU(U, 2)
        Ugood = [0, 0.5, 1]
        np.testing.assert_allclose(Utest, Ugood)

        Utest = SUF.rafineU(U, 4)
        Ugood = [0, 0.25, 0.5, 0.75, 1]
        np.testing.assert_allclose(Utest, Ugood)

        Utest = SUF.rafineU(U, 5)
        Ugood = [0, 0.2, 0.4, 0.6, 0.8, 1]
        np.testing.assert_allclose(Utest, Ugood)

    if True:
        U = SUF.UBezier(p=2)

        Utest = SUF.rafineU(U, 1)
        Ugood = [0, 1]
        np.testing.assert_allclose(Utest, Ugood)

        Utest = SUF.rafineU(U, 2)
        Ugood = [0, 0.5, 1]
        np.testing.assert_allclose(Utest, Ugood)

        Utest = SUF.rafineU(U, 4)
        Ugood = [0, 0.25, 0.5, 0.75, 1]
        np.testing.assert_allclose(Utest, Ugood)

        Utest = SUF.rafineU(U, 5)
        Ugood = [0, 0.2, 0.4, 0.6, 0.8, 1]
        np.testing.assert_allclose(Utest, Ugood)

    if True:
        U = SUF.UUniform(p=1, n=3)

        Utest = SUF.rafineU(U, 1)
        Ugood = [0, 0.5, 1]
        np.testing.assert_allclose(Utest, Ugood)

        Utest = SUF.rafineU(U, 2)
        Ugood = [0, 0.25, 0.5, 0.75, 1]
        np.testing.assert_allclose(Utest, Ugood)

        Utest = SUF.rafineU(U, 4)
        Ugood = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
        np.testing.assert_allclose(Utest, Ugood)

        Utest = SUF.rafineU(U, 5)
        Ugood = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        np.testing.assert_allclose(Utest, Ugood)
