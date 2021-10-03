import numpy as np
from numpy import linalg as la
import pytest
import femnurbs.SplineUsefulFunctions as SUF
import femnurbs.SolveHeatEquation1D as SHE


def test_SumMassMatrixValue():
    Ntest = 100
    for p in (1, 2, 3):
        for zz in range(Ntest):
            for n in range(2, 10):
                U = SUF.URandom(p=p, n=n)
                Mtest = SHE.getMassMatrix(U)
                assert np.sum(Mtest) == pytest.approx(0)


def test_MassMatrixWhenP1():
    U = SUF.UBezier(p=1)  # n = 2
    Mgood = np.array([[1, -1],
                      [-1, 1]])
    Mtest = SHE.getMassMatrix(U)
    np.testing.assert_allclose(Mgood, Mtest)

    p, n = 1, 3
    U = SUF.UUniform(p=p, n=n)
    Mgood = 2 * np.array([[1, -1, 0],
                          [-1, 2, -1],
                          [0, -1, 1]])
    Mtest = SHE.getMassMatrix(U)
    np.testing.assert_allclose(Mgood, Mtest)

    p, n = 1, 4
    U = SUF.UUniform(p=p, n=n)
    Mgood = 3 * np.array([[1, -1, 0, 0],
                          [-1, 2, -1, 0],
                          [0, -1, 2, -1],
                          [0, 0, -1, 1]])
    Mtest = SHE.getMassMatrix(U)
    np.testing.assert_allclose(Mgood, Mtest)


def test_MassMatrixWhenP2():
    U = SUF.UBezier(p=2)  # n = 3
    Mgood = (2 / 3) * np.array([[2, -1, -1],
                                [-1, 2, -1],
                                [-1, -1, 2]])
    Mtest = SHE.getMassMatrix(U)
    np.testing.assert_allclose(Mgood, Mtest)

    p, n = 2, 4
    U = SUF.UUniform(p=p, n=n)
    Mgood = (2 / 3) * np.array([[4, -3, -1, 0],
                                [-3, 4, 0, -1],
                                [-1, 0, 4, -3],
                                [0, -1, -3, 4]])
    Mtest = SHE.getMassMatrix(U)
    np.testing.assert_allclose(Mgood, Mtest)

    p, n = 2, 5
    U = SUF.UUniform(p=p, n=n)
    Mgood = (1 / 2) * np.array([[8, -6, -2, 0, 0],
                                [-6, 8, -1, -1, 0],
                                [-2, -1, 6, -1, -2],
                                [0, -1, -1, 8, -6],
                                [0, 0, -2, -6, 8]])
    Mtest = SHE.getMassMatrix(U)
    np.testing.assert_allclose(Mgood, Mtest)


def test_MassMatrixWhenP3():
    U = SUF.UBezier(p=3)  # n = 4
    Mgood = (3 / 10) * np.array([[6, -3, -2, -1],
                                 [-3, 4, 1, -2],
                                 [-2, 1, 4, -3],
                                 [-1, -2, -3, 6]])
    Mtest = SHE.getMassMatrix(U)
    np.testing.assert_allclose(Mgood, Mtest)

    p, n = 3, 5
    U = SUF.UUniform(p=p, n=n)
    Mgood = (3 / 20) * np.array([[24, -17, -6, -1, 0],
                                 [-17, 20, 2, -4, -1],
                                 [-6, 2, 8, 2, -6],
                                 [-1, -4, 2, 20, -17],
                                 [0, -1, -6, -17, 24]])
    Mtest = SHE.getMassMatrix(U)
    np.testing.assert_allclose(Mgood, Mtest)

    p, n = 3, 6
    U = SUF.UUniform(p=p, n=n)
    Mgood = (3 / 160) * np.array([[288, -204, -76, -8, 0, 0],
                                  [-204, 240, 6, -39, -3, 0],
                                  [-76, 6, 108, 9, -39, -8],
                                  [-8, -39, 9, 108, 6, -76],
                                  [0, -3, -39, 6, 240, -204],
                                  [0, 0, -8, -76, -204, 288]])
    Mtest = SHE.getMassMatrix(U)
    np.testing.assert_allclose(Mgood, Mtest)


def test_ForceVectorP1():
    def f0(u):
        return 0

    def f1(u):
        return 1

    def fu(u):
        return u

    def fu2(u):
        return u**2

    p, n = 1, 2
    U = SUF.UBezier(p=1)  # n = 2
    Fgood = np.zeros(n)
    Ftest = SHE.getForceVector(U, f0)
    np.testing.assert_allclose(Fgood, Ftest)

    p, n = 1, 3
    U = SUF.UUniform(p=p, n=n)
    Fgood = np.zeros(n)
    Ftest = SHE.getForceVector(U, f0)
    np.testing.assert_allclose(Fgood, Ftest)

    p, n = 1, 4
    U = SUF.UUniform(p=p, n=n)
    Fgood = np.zeros(n)
    Ftest = SHE.getForceVector(U, f0)
    np.testing.assert_allclose(Fgood, Ftest)

    p, n = 1, 2
    U = SUF.UBezier(p=1)  # n = 2
    Fgood = np.ones(n) / 2
    Ftest = SHE.getForceVector(U, f1)
    np.testing.assert_allclose(Fgood, Ftest)

    p, n = 1, 3
    U = SUF.UUniform(p=p, n=n)
    Fgood = np.array([1, 2, 1]) / 4
    Ftest = SHE.getForceVector(U, f1)
    np.testing.assert_allclose(Fgood, Ftest)

    p, n = 1, 4
    U = SUF.UUniform(p=p, n=n)
    Fgood = np.array([1, 2, 2, 1]) / 6
    Ftest = SHE.getForceVector(U, f1)
    np.testing.assert_allclose(Fgood, Ftest)

    for n in range(2, 10):
        U = SUF.URandom(p=1, n=n)

        Fgood = np.zeros(n)
        Ftest = SHE.getForceVector(U, f0)
        np.testing.assert_allclose(Fgood, Ftest)

        Ftest = SHE.getForceVector(U, f1)
        assert np.sum(Ftest) == pytest.approx(1)

def test_solutionForceNull():

    def force(u):
        return 0

    Tl, Tr = 0, 0
    for p in (1, 2, 3):
        for n in range(p + 1, 10):
            U = SUF.UUniform(p=p, n=n)
            Tgood = np.zeros(n)
            Ttest = SHE.findSolution(U, force, (Tl, Tr))
            np.testing.assert_allclose(Tgood, Ttest)

    Tl, Tr = 1, 1
    for p in (1, 2, 3):
        for n in range(p + 1, 10):
            U = SUF.UUniform(p=p, n=n)
            Tgood = np.ones(n)
            Ttest = SHE.findSolution(U, force, (Tl, Tr))
            np.testing.assert_allclose(Tgood, Ttest)

    p, n = 1, 3
    Tl, Tr = 0, 1
    U = SUF.UUniform(p=p, n=n)
    Tgood = np.array([0, 0.5, 1])
    Ttest = SHE.findSolution(U, force, (Tl, Tr))
    np.testing.assert_allclose(Tgood, Ttest)

    p, n = 2, 6
    Tl, Tr = 1, 0
    U = SUF.UUniform(p=p, n=n)
    Tgood = np.array([1, 0.875, 0.625, 0.375, 0.125, 0])
    Ttest = SHE.findSolution(U, force, (Tl, Tr))
    np.testing.assert_allclose(Tgood, Ttest)

    p, n = 2, 6
    Tl, Tr = 0, 1
    U = SUF.UUniform(p=p, n=n)
    Tgood = np.array([0, 0.125, 0.375, 0.625, 0.875, 1])
    Ttest = SHE.findSolution(U, force, (Tl, Tr))
    np.testing.assert_allclose(Tgood, Ttest)

def test_solutionForceConstant1():

    def force(u):
        return 1

    p, n = 1, 3
    Tl, Tr = 0, 0
    U = SUF.UUniform(p=p, n=n)
    Tgood = np.array([0, 1, 0]) / 8
    Ttest = SHE.findSolution(U, force, (Tl, Tr))
    np.testing.assert_allclose(Tgood, Ttest)

    p, n = 1, 4
    Tl, Tr = 1, 0
    U = SUF.UUniform(p=p, n=n)
    Tgood = np.array([9, 7, 4, 0]) / 9
    Ttest = SHE.findSolution(U, force, (Tl, Tr))
    np.testing.assert_allclose(Tgood, Ttest)

    p, n = 1, 5
    Tl, Tr = 1, 0
    U = SUF.UUniform(p=p, n=n)
    Tgood = np.array([32, 27, 20, 11, 0]) / 32
    Ttest = SHE.findSolution(U, force, (Tl, Tr))
    np.testing.assert_allclose(Tgood, Ttest)

    p, n = 2, 3
    Tl, Tr = 1, 0
    U = SUF.UUniform(p=p, n=n)
    Tgood = np.array([4, 3, 0]) / 4
    Ttest = SHE.findSolution(U, force, (Tl, Tr))
    np.testing.assert_allclose(Tgood, Ttest)

    p, n = 2, 4
    Tl, Tr = 1, 0
    U = SUF.UUniform(p=p, n=n)
    Tgood = np.array([8, 7, 3, 0]) / 8
    Ttest = SHE.findSolution(U, force, (Tl, Tr))
    np.testing.assert_allclose(Tgood, Ttest)

    p, n = 2, 5
    Tl, Tr = 1, 0
    U = SUF.UUniform(p=p, n=n)
    Tgood = np.array([36, 33, 23, 9, 0]) / 36
    Ttest = SHE.findSolution(U, force, (Tl, Tr))
    np.testing.assert_allclose(Tgood, Ttest)

    p, n = 3, 4
    Tl, Tr = 1, 0
    U = SUF.UUniform(p=p, n=n)
    Tgood = np.array([6, 5, 3, 0]) / 6
    Ttest = SHE.findSolution(U, force, (Tl, Tr))
    np.testing.assert_allclose(Tgood, Ttest)

    p, n = 3, 5
    Tl, Tr = 1, 0
    U = SUF.UUniform(p=p, n=n)
    Tgood = np.array([12, 11, 8, 3, 0]) / 12
    Ttest = SHE.findSolution(U, force, (Tl, Tr))
    np.testing.assert_allclose(Tgood, Ttest)

def test_solutionForceConstant2():

    def force(u):
        return 2

    p, n = 1, 3
    Tl, Tr = 0, 0
    U = SUF.UUniform(p=p, n=n)
    Tgood = np.array([0, 1, 0]) / 4
    Ttest = SHE.findSolution(U, force, (Tl, Tr))
    np.testing.assert_allclose(Tgood, Ttest)

    p, n = 1, 4
    Tl, Tr = 0, 0
    U = SUF.UUniform(p=p, n=n)
    Tgood = np.array([0, 2, 2, 0]) / 9
    Ttest = SHE.findSolution(U, force, (Tl, Tr))
    np.testing.assert_allclose(Tgood, Ttest)

    p, n = 1, 5
    Tl, Tr = 0, 0
    U = SUF.UUniform(p=p, n=n)
    Tgood = np.array([0, 3, 4, 3, 0]) / 16
    Ttest = SHE.findSolution(U, force, (Tl, Tr))
    np.testing.assert_allclose(Tgood, Ttest)

    p, n = 2, 3
    Tl, Tr = 0, 0
    U = SUF.UUniform(p=p, n=n)
    Tgood = np.array([0, 1, 0]) / 2
    Ttest = SHE.findSolution(U, force, (Tl, Tr))
    np.testing.assert_allclose(Tgood, Ttest)

    p, n = 2, 4
    Tl, Tr = 0, 0
    U = SUF.UUniform(p=p, n=n)
    Tgood = np.array([0, 1, 1, 0]) / 4
    Ttest = SHE.findSolution(U, force, (Tl, Tr))
    np.testing.assert_allclose(Tgood, Ttest)

    p, n = 2, 5
    Tl, Tr = 0, 0
    U = SUF.UUniform(p=p, n=n)
    Tgood = np.array([0, 3, 5, 3, 0]) / 18
    Ttest = SHE.findSolution(U, force, (Tl, Tr))
    np.testing.assert_allclose(Tgood, Ttest)

    p, n = 3, 4
    Tl, Tr = 0, 0
    U = SUF.UUniform(p=p, n=n)
    Tgood = np.array([0, 1, 1, 0]) / 3
    Ttest = SHE.findSolution(U, force, (Tl, Tr))
    np.testing.assert_allclose(Tgood, Ttest)

    p, n = 3, 5
    Tl, Tr = 0, 0
    U = SUF.UUniform(p=p, n=n)
    Tgood = np.array([0, 1, 2, 1, 0]) / 6
    Ttest = SHE.findSolution(U, force, (Tl, Tr))
    np.testing.assert_allclose(Tgood, Ttest)

def test_solutionForcePar():

    def force1(u):
        return (2 * u - 1)**2

    def force2(u):
        return np.cos(2 * u - 1)

    Tl, Tr = 0, 0
    for force in (force1, force2):
        for p in (1, 2, 3):
            for n in range(p + 1, 6):
                U = SUF.UUniform(p=p, n=n)
                Ttest = SHE.findSolution(U, force, (Tl, Tr))
                assert SUF.isSymetric(Ttest) is True


def test_solutionForceImpar():
    def force1(u):
        return (2 * u - 1)

    def force2(u):
        return (2 * u - 1)**3

    def force3(u):
        return np.sin(2 * u - 1)

    def force4(u):
        return np.tan(2 * u - 1)

    Tl, Tr = 0, 0
    for force in (force1, force2, force3, force4):
        for p in (1, 2, 3):
            for n in range(p + 1, 6):
                U = SUF.UUniform(p=p, n=n)
                Ttest = SHE.findSolution(U, force, (Tl, Tr))
                assert np.sum(Ttest) == pytest.approx(0)
