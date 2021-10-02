import pytest
import numpy as np
import femnurbs.SplineSimpleIntegral as SSI
import femnurbs.SplineUsefulFunctions as SUF


def test_V0():
    Vtest = np.array([1])
    Vresult = SSI.SplineIntegralSimple.getIntegralBase(p=0)
    np.testing.assert_almost_equal(Vresult, Vtest)


def test_V1():
    Vtest = np.array([1, 1]) / 2
    Vresult = SSI.SplineIntegralSimple.getIntegralBase(p=1)
    np.testing.assert_almost_equal(Vresult, Vtest)


def test_V2():
    sides = np.array([[1], [1]])
    Vtest = np.array([1, 4, 1]) / 6
    Vresult = SSI.SplineIntegralSimple.getIntegralBase(p=2, sides=sides)
    np.testing.assert_almost_equal(Vresult, Vtest)

    sides = np.array([[0], [1]])
    Vtest = np.array([2, 3, 1]) / 6
    Vresult = SSI.SplineIntegralSimple.getIntegralBase(p=2, sides=sides)
    np.testing.assert_almost_equal(Vresult, Vtest)

    sides = np.array([[1], [0]])
    Vtest = np.array([1, 3, 2]) / 6
    Vresult = SSI.SplineIntegralSimple.getIntegralBase(p=2, sides=sides)
    np.testing.assert_almost_equal(Vresult, Vtest)

    sides = np.array([[0], [1 / 2]])
    Vtest = np.array([3, 4, 2]) / 9
    Vresult = SSI.SplineIntegralSimple.getIntegralBase(p=2, sides=sides)
    np.testing.assert_almost_equal(Vresult, Vtest)

    sides = np.array([[1 / 2], [1 / 2]])
    Vtest = np.array([2, 5, 2]) / 9
    Vresult = SSI.SplineIntegralSimple.getIntegralBase(p=2, sides=sides)
    np.testing.assert_almost_equal(Vresult, Vtest)

    sides = np.array([[1 / 2], [0]])
    Vtest = np.array([2, 4, 3]) / 9
    Vresult = SSI.SplineIntegralSimple.getIntegralBase(p=2, sides=sides)
    np.testing.assert_almost_equal(Vresult, Vtest)


def test_V3():
    sides = np.array([[1, 1], [1, 1]])
    Vtest = np.array([1, 11, 11, 1]) / 24
    Vresult = SSI.SplineIntegralSimple.getIntegralBase(p=3, sides=sides)
    np.testing.assert_almost_equal(Vresult, Vtest)

    sides = np.array([[1, 0], [1, 1]])
    Vtest = np.array([3, 21, 22, 2]) / 48
    Vresult = SSI.SplineIntegralSimple.getIntegralBase(p=3, sides=sides)
    np.testing.assert_almost_equal(Vresult, Vtest)

    sides = np.array([[0, 0], [1, 1]])
    Vtest = np.array([12, 21, 13, 2]) / 48
    Vresult = SSI.SplineIntegralSimple.getIntegralBase(p=3, sides=sides)
    np.testing.assert_almost_equal(Vresult, Vtest)

    sides = np.array([[0, 0], [0, 1]])
    Vtest = np.array([2, 2, 3, 1]) / 8
    Vresult = SSI.SplineIntegralSimple.getIntegralBase(p=3, sides=sides)
    np.testing.assert_almost_equal(Vresult, Vtest)

    sides = np.array([[0, 0], [0, 0]])
    Vtest = np.array([1, 1, 1, 1]) / 4
    Vresult = SSI.SplineIntegralSimple.getIntegralBase(p=3, sides=sides)
    np.testing.assert_almost_equal(Vresult, Vtest)

    sides = np.array([[0, 1], [0, 0]])
    Vtest = np.array([1, 3, 2, 2]) / 8
    Vresult = SSI.SplineIntegralSimple.getIntegralBase(p=3, sides=sides)
    np.testing.assert_almost_equal(Vresult, Vtest)

    sides = np.array([[1, 1], [0, 0]])
    Vtest = np.array([2, 13, 21, 12]) / 48
    Vresult = SSI.SplineIntegralSimple.getIntegralBase(p=3, sides=sides)
    np.testing.assert_almost_equal(Vresult, Vtest)

    sides = np.array([[1, 1], [1, 0]])
    Vtest = np.array([2, 22, 21, 3]) / 48
    Vresult = SSI.SplineIntegralSimple.getIntegralBase(p=3, sides=sides)
    np.testing.assert_almost_equal(Vresult, Vtest)

    sides = np.array([[0.2, 0.2], [0.2, 0.2]])
    Vtest = np.array([25, 59, 59, 25]) / 168
    Vresult = SSI.SplineIntegralSimple.getIntegralBase(p=3, sides=sides)
    np.testing.assert_almost_equal(Vresult, Vtest)


def test_VectorH():
    U = SUF.UBezier(p=0)
    Hresult = SSI.SplineIntegralSimple.getVectorH(U)
    Htest = np.array([1])
    np.testing.assert_almost_equal(Hresult, Htest)

    U = SUF.UBezier(p=1)
    Hresult = SSI.SplineIntegralSimple.getVectorH(U)
    Htest = np.array([1])
    np.testing.assert_almost_equal(Hresult, Htest)

    U = SUF.UBezier(p=2)
    Hresult = SSI.SplineIntegralSimple.getVectorH(U)
    Htest = np.array([0, 1, 0])
    np.testing.assert_almost_equal(Hresult, Htest)

    U = SUF.UBezier(p=3)
    Hresult = SSI.SplineIntegralSimple.getVectorH(U)
    Htest = np.array([0, 0, 1, 0, 0])
    np.testing.assert_almost_equal(Hresult, Htest)

    U = SUF.UUniform(p=0, n=2)
    Hresult = SSI.SplineIntegralSimple.getVectorH(U)
    Htest = np.array([0.5, 0.5])
    np.testing.assert_almost_equal(Hresult, Htest)

    U = SUF.UUniform(p=1, n=3)
    Hresult = SSI.SplineIntegralSimple.getVectorH(U)
    Htest = np.array([0.5, 0.5])
    np.testing.assert_almost_equal(Hresult, Htest)

    U = SUF.UUniform(p=2, n=4)
    Hresult = SSI.SplineIntegralSimple.getVectorH(U)
    Htest = np.array([0, 0.5, 0.5, 0])
    np.testing.assert_almost_equal(Hresult, Htest)

    U = SUF.UUniform(p=3, n=5)
    Hresult = SSI.SplineIntegralSimple.getVectorH(U)
    Htest = np.array([0, 0, 0.5, 0.5, 0, 0])
    np.testing.assert_almost_equal(Hresult, Htest)

    U = SUF.UUniform(p=0, n=5)
    Hresult = SSI.SplineIntegralSimple.getVectorH(U)
    Htest = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    np.testing.assert_almost_equal(Hresult, Htest)

    U = SUF.UUniform(p=1, n=6)
    Hresult = SSI.SplineIntegralSimple.getVectorH(U)
    Htest = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    np.testing.assert_almost_equal(Hresult, Htest)

    U = SUF.UUniform(p=2, n=7)
    Hresult = SSI.SplineIntegralSimple.getVectorH(U)
    Htest = np.array([0, 0.2, 0.2, 0.2, 0.2, 0.2, 0])
    np.testing.assert_almost_equal(Hresult, Htest)

    U = SUF.UUniform(p=3, n=8)
    Hresult = SSI.SplineIntegralSimple.getVectorH(U)
    Htest = np.array([0, 0, 0.2, 0.2, 0.2, 0.2, 0.2, 0, 0])
    np.testing.assert_almost_equal(Hresult, Htest)


def test_transformHpartedIntoSizes():
    Htest = np.array([1])
    sizesTest = np.array([[], []])
    sizesResult = SSI.SplineIntegralSimple.transformHintoSizes(Htest)
    np.testing.assert_array_equal(sizesResult, sizesTest)

    Htest = np.array([1, 1, 1])
    sizesTest = np.array([[1], [1]])
    sizesResult = SSI.SplineIntegralSimple.transformHintoSizes(Htest)
    np.testing.assert_array_equal(sizesResult, sizesTest)

    Htest = np.array([1, 2, 1])
    sizesTest = np.array([[0.5], [0.5]])
    sizesResult = SSI.SplineIntegralSimple.transformHintoSizes(Htest)
    np.testing.assert_array_equal(sizesResult, sizesTest)

    Htest = np.array([2, 1, 2])
    sizesTest = np.array([[2], [2]])
    sizesResult = SSI.SplineIntegralSimple.transformHintoSizes(Htest)
    np.testing.assert_array_equal(sizesResult, sizesTest)

    Htest = np.array([1, 1, 1, 1, 1])
    sizesTest = np.array([[1, 1], [1, 1]])
    sizesResult = SSI.SplineIntegralSimple.transformHintoSizes(Htest)
    np.testing.assert_array_equal(sizesResult, sizesTest)

    Htest = np.array([1, 1, 2, 1, 1])
    sizesTest = np.array([[0.5, 0.5], [0.5, 0.5]])
    sizesResult = SSI.SplineIntegralSimple.transformHintoSizes(Htest)
    np.testing.assert_array_equal(sizesResult, sizesTest)

    Htest = np.array([2, 2, 1, 2, 2])
    sizesTest = np.array([[2, 2], [2, 2]])
    sizesResult = SSI.SplineIntegralSimple.transformHintoSizes(Htest)
    np.testing.assert_array_equal(sizesResult, sizesTest)

    Htest = np.array([2, 2, 1, 1, 1])
    sizesTest = np.array([[2, 2], [1, 1]])
    sizesResult = SSI.SplineIntegralSimple.transformHintoSizes(Htest)
    np.testing.assert_array_equal(sizesResult, sizesTest)

    Htest = np.array([1, 1, 1, 2, 2])
    sizesTest = np.array([[1, 1], [2, 2]])
    sizesResult = SSI.SplineIntegralSimple.transformHintoSizes(Htest)
    np.testing.assert_array_equal(sizesResult, sizesTest)

    Htest = np.array([3, 2, 1, 2, 3])
    sizesTest = np.array([[2, 3], [2, 3]])
    sizesResult = SSI.SplineIntegralSimple.transformHintoSizes(Htest)
    np.testing.assert_array_equal(sizesResult, sizesTest)


def test_IntegralAllDomain():
    U = SUF.UBezier(p=1)
    Vtest = SSI.SplineIntegralSimple.getIntegralAllDomain(U)
    Vresult = np.array([1, 1]) / 2
    np.testing.assert_almost_equal(Vresult, Vtest)

    U = SUF.UBezier(p=2)
    Vtest = SSI.SplineIntegralSimple.getIntegralAllDomain(U)
    Vresult = np.array([1, 1, 1]) / 3
    np.testing.assert_almost_equal(Vresult, Vtest)

    U = SUF.UBezier(p=3)
    Vtest = SSI.SplineIntegralSimple.getIntegralAllDomain(U)
    Vresult = np.array([1, 1, 1, 1]) / 4
    np.testing.assert_almost_equal(Vresult, Vtest)

    U = SUF.UUniform(p=1, n=3)
    Vtest = SSI.SplineIntegralSimple.getIntegralAllDomain(U)
    Vresult = np.array([1, 2, 1]) / 4
    np.testing.assert_almost_equal(Vresult, Vtest)

    U = SUF.UUniform(p=2, n=4)
    Vtest = SSI.SplineIntegralSimple.getIntegralAllDomain(U)
    Vresult = np.array([1, 2, 2, 1]) / 6
    np.testing.assert_almost_equal(Vresult, Vtest)

    U = SUF.UUniform(p=3, n=5)
    Vtest = SSI.SplineIntegralSimple.getIntegralAllDomain(U)
    Vresult = np.array([1, 2, 2, 2, 1]) / 8
    np.testing.assert_almost_equal(Vresult, Vtest)

    U = SUF.UUniform(p=1, n=6)
    Vtest = SSI.SplineIntegralSimple.getIntegralAllDomain(U)
    Vresult = np.array([1, 2, 2, 2, 2, 1]) / 10
    np.testing.assert_almost_equal(Vresult, Vtest)

    U = SUF.UUniform(p=2, n=7)
    Vtest = SSI.SplineIntegralSimple.getIntegralAllDomain(U)
    Vresult = np.array([1, 2, 3, 3, 3, 2, 1]) / 15
    np.testing.assert_almost_equal(Vresult, Vtest)

    U = SUF.UUniform(p=3, n=8)
    Vtest = SSI.SplineIntegralSimple.getIntegralAllDomain(U)
    Vresult = np.array([1, 2, 3, 4, 4, 3, 2, 1]) / 20
    np.testing.assert_almost_equal(Vresult, Vtest)

    U = SUF.UUniform(p=2, n=6)
    Vtest = SSI.SplineIntegralSimple.getIntegralAllDomain(U)
    Vresult = np.array([1, 2, 3, 3, 2, 1]) / 12
    np.testing.assert_almost_equal(Vresult, Vtest)

    U = SUF.UUniform(p=3, n=6)
    Vtest = SSI.SplineIntegralSimple.getIntegralAllDomain(U)
    Vresult = np.array([1, 2, 3, 3, 2, 1]) / 12
    np.testing.assert_almost_equal(Vresult, Vtest)
