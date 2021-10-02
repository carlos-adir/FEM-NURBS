import pytest
import numpy as np
import femnurbs.SplineDoubleIntegral as SDI
import femnurbs.SplineUsefulFunctions as SUF


def test_M00Shapes():
    M = SDI.SplineIntegralDouble.getIntegralBase(j=0, k=0)
    np.testing.assert_array_equal(M.shape, (1, 1))


def test_M00SumAllValues():
    M = SDI.SplineIntegralDouble.getIntegralBase(j=0, k=0)
    assert np.sum(M) == pytest.approx(1)


def test_M00Values():
    Mgood = np.eye(1)
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=0, k=0)
    np.testing.assert_almost_equal(Mgood, Mtest)


def test_M10Shapes():
    M = SDI.SplineIntegralDouble.getIntegralBase(j=1, k=0)
    np.testing.assert_array_equal(M.shape, (2, 1))


def test_M10SumAllValues():
    M = SDI.SplineIntegralDouble.getIntegralBase(j=1, k=0)
    assert np.sum(M) == pytest.approx(1)


def test_M10Values():
    Mgood = np.ones((2, 1)) / 2
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=1, k=0)
    np.testing.assert_almost_equal(Mgood, Mtest)


def test_M11Shapes():
    M = SDI.SplineIntegralDouble.getIntegralBase(j=1, k=1)
    np.testing.assert_array_equal(M.shape, (2, 2))


def test_M11SumAllValues():
    M = SDI.SplineIntegralDouble.getIntegralBase(j=1, k=1)
    assert np.sum(M) == pytest.approx(1)


def test_M11Values():
    Mgood = np.array([[2, 1],
                      [1, 2]]) / 6
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=1, k=1)
    np.testing.assert_almost_equal(Mgood, Mtest)


def test_M20Shapes():
    NumberTest = 100
    for i in range(NumberTest):
        sides = 2 * np.random.rand(2, 1)
        M = SDI.SplineIntegralDouble.getIntegralBase(j=2, k=0, sides=sides)
        np.testing.assert_array_equal(M.shape, (3, 1))


def test_M20SumAllValues():
    NumberTest = 100
    for i in range(NumberTest):
        sides = 2 * np.random.rand(2, 1)
        M = SDI.SplineIntegralDouble.getIntegralBase(j=2, k=0, sides=sides)
        assert np.sum(M) == pytest.approx(1)


def test_M20Values():
    sides = np.array([[1], [1]])
    Mgood = np.array([[1, 4, 1]]).T / 6
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=2, k=0, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[0], [1]])
    Mgood = np.array([[2, 3, 1]]).T / 6
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=2, k=0, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[1], [0]])
    Mgood = np.array([[1, 3, 2]]).T / 6
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=2, k=0, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)


def test_M21Shapes():
    NumberTest = 100
    for i in range(NumberTest):
        sides = 2 * np.random.rand(2, 1)
        M = SDI.SplineIntegralDouble.getIntegralBase(j=2, k=1, sides=sides)
        np.testing.assert_array_equal(M.shape, (3, 2))


def test_M21SumAllValues():
    NumberTest = 100
    for i in range(NumberTest):
        sides = 2 * np.random.rand(2, 1)
        M = SDI.SplineIntegralDouble.getIntegralBase(j=2, k=1, sides=sides)
        assert np.sum(M) == pytest.approx(1)


def test_M21Values():
    sides = np.array([[1], [1]])
    Mgood = np.array([[3, 1],
                      [8, 8],
                      [1, 3]]) / 24
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=2, k=1, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[1], [0]])
    Mgood = np.array([[3, 1],
                      [7, 5],
                      [2, 6]]) / 24
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=2, k=1, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[0], [0]])
    Mgood = np.array([[3, 1],
                      [2, 2],
                      [1, 3]]) / 12
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=2, k=1, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[0], [1]])
    Mgood = np.array([[6, 2],
                      [5, 7],
                      [1, 3]]) / 24
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=2, k=1, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)


def test_M22Shapes():
    NumberTest = 100
    for i in range(NumberTest):
        sides = 2 * np.random.rand(2, 1)
        M = SDI.SplineIntegralDouble.getIntegralBase(j=2, k=2, sides=sides)
        np.testing.assert_array_equal(M.shape, (3, 3))


def test_M22SymetryMatrix():
    NumberTest = 100
    for i in range(NumberTest):
        sides = 2 * np.random.rand(2, 1)
        M = SDI.SplineIntegralDouble.getIntegralBase(j=2, k=2, sides=sides)
        assert SUF.isSymetric(M) is True


def test_M22SymetrySides():
    NumberTest = 100
    for i in range(NumberTest):
        side0 = 2 * np.random.rand(1)
        sides = np.array([side0, side0])
        M = SDI.SplineIntegralDouble.getIntegralBase(j=2, k=2, sides=sides)
        assert SUF.isSymetric(M, diagonal=2) is True


def test_M22SumAllValues():
    NumberTest = 100
    for i in range(NumberTest):
        sides = 2 * np.random.rand(2, 1)
        M = SDI.SplineIntegralDouble.getIntegralBase(j=2, k=2, sides=sides)
        assert np.sum(M) == pytest.approx(1)


def test_M22Values():
    sides = np.array([[1], [1]])
    Mgood = np.array([[6, 13, 1],
                      [13, 54, 13],
                      [1, 13, 6]]) / 120
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=2, k=2, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[0], [1]])
    Mgood = np.array([[12, 7, 1],
                      [7, 17, 6],
                      [1, 6, 3]]) / 60
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=2, k=2, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[0], [0]])
    Mgood = np.array([[6, 3, 1],
                      [3, 4, 3],
                      [1, 3, 6]]) / 30
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=2, k=2, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[0], [0.5]])
    Mgood = np.array([[9, 5, 1],
                      [5, 10, 5],
                      [1, 5, 4]]) / 45
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=2, k=2, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[0.5], [0.5]])
    Mgood = np.array([[12, 16, 2],
                      [16, 43, 16],
                      [2, 16, 12]]) / 135
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=2, k=2, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[0.5], [1]])
    Mgood = np.array([[16, 22, 2],
                      [22, 69, 19],
                      [2, 19, 9]]) / 180
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=2, k=2, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[1 / 3], [1]])
    Mgood = np.array([[27, 30, 3],
                      [30, 85, 25],
                      [3, 25, 12]]) / 240
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=2, k=2, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[2 / 3], [1]])
    Mgood = np.array([[108, 177, 15],
                      [177, 613, 160],
                      [15, 160, 75]]) / 1500
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=2, k=2, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[1 / 3], [1 / 3]])
    Mgood = np.array([[18, 19, 3],
                      [19, 42, 19],
                      [3, 19, 18]]) / 160
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=2, k=2, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[2 / 3], [2 / 3]])
    Mgood = np.array([[18, 29, 3],
                      [29, 92, 29],
                      [3, 29, 18]]) / 250
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=2, k=2, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)


def test_M30Shapes():
    NumberTest = 100
    for i in range(NumberTest):
        sides = 2 * np.random.rand(2, 2)
        M = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=0, sides=sides)
        np.testing.assert_array_equal(M.shape, (4, 1))


def test_M30SumAllValues():
    NumberTest = 100
    for i in range(NumberTest):
        sides = 2 * np.random.rand(2, 2)
        M = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=0, sides=sides)
        assert np.sum(M) == pytest.approx(1)


def test_M30Values():
    sides = np.array([[1, 1], [1, 1]])
    Mgood = np.array([[1, 11, 11, 1]]).T / 24
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=0, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[1, 0], [1, 1]])
    Mgood = np.array([[3, 21, 22, 2]]).T / 48
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=0, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[0, 0], [1, 1]])
    Mgood = np.array([[12, 21, 13, 2]]).T / 48
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=0, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[0, 0], [1, 0]])
    Mgood = np.array([[4, 7, 4, 1]]).T / 16
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=0, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[0, 0], [0, 0]])
    Mgood = np.array([[1, 1, 1, 1]]).T / 4
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=0, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[1, 0], [1, 0]])
    Mgood = np.array([[1, 7, 7, 1]]).T / 16
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=0, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[0, 1], [0, 1]])
    Mgood = np.array([[1, 3, 3, 1]]).T / 8
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=0, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)


def test_M31Shapes():
    NumberTest = 100
    for i in range(NumberTest):
        sides = 2 * np.random.rand(2, 2)
        M = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=1, sides=sides)
        np.testing.assert_array_equal(M.shape, (4, 2))


def test_M31SumAllValues():
    NumberTest = 100
    for i in range(NumberTest):
        sides = 2 * np.random.rand(2, 2)
        M = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=1, sides=sides)
        assert np.sum(M) == pytest.approx(1)


def test_M31Values():
    sides = np.array([[1, 1], [1, 1]])
    Mgood = np.array([[4, 33, 22, 1],
                      [1, 22, 33, 4]]).T / 120
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=1, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[1, 0], [1, 1]])
    Mgood = np.array([[12, 62, 44, 2],
                      [3, 43, 66, 8]]).T / 240
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=1, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[0, 0], [1, 1]])
    Mgood = np.array([[48, 51, 19, 2],
                      [12, 54, 46, 8]]).T / 240
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=1, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[0, 0], [1, 0]])
    Mgood = np.array([[16, 17, 6, 1],
                      [4, 18, 14, 4]]).T / 80
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=1, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[0, 0], [0, 0]])
    Mgood = np.array([[4, 3, 2, 1],
                      [1, 2, 3, 4]]).T / 20
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=1, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[1, 0], [1, 0]])
    Mgood = np.array([[12, 62, 43, 3],
                      [3, 43, 62, 12]]).T / 240
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=1, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[0, 1], [0, 1]])
    Mgood = np.array([[4, 10, 5, 1],
                      [1, 5, 10, 4]]).T / 40
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=1, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)


def test_M32Shapes():
    NumberTest = 100
    for i in range(NumberTest):
        sides = 2 * np.random.rand(2, 2)
        M = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=2, sides=sides)
        np.testing.assert_array_equal(M.shape, (4, 3))


def test_M32SumAllValues():
    NumberTest = 100
    for i in range(NumberTest):
        sides = 2 * np.random.rand(2, 2)
        M = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=2, sides=sides)
        assert np.sum(M) == pytest.approx(1)


def test_M32Values():
    sides = np.array([[1, 1], [1, 1]])
    Mgood = np.array([[10, 71, 38, 1],
                      [19, 221, 221, 19],
                      [1, 38, 71, 10]]).T / 720
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=2, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[1, 0], [1, 1]])
    Mgood = np.array([[30, 132, 76, 2],
                      [57, 423, 442, 38],
                      [3, 75, 142, 20]]).T / 1440
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=2, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[0, 0], [1, 1]])
    Mgood = np.array([[120, 93, 25, 2],
                      [54, 171, 117, 18],
                      [6, 51, 53, 10]]).T / 720
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=2, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[0, 0], [1, 0]])
    Mgood = np.array([[40, 31, 8, 1],
                      [18, 57, 36, 9],
                      [2, 17, 16, 5]]).T / 240
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=2, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[0, 0], [0, 0]])
    Mgood = np.array([[10, 6, 3, 1],
                      [4, 6, 6, 4],
                      [1, 3, 6, 10]]).T / 60
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=2, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[1, 0], [1, 0]])
    Mgood = np.array([[10, 44, 25, 1],
                      [19, 141, 141, 19],
                      [1, 25, 44, 10]]).T / 480
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=2, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[0, 1], [0, 1]])
    Mgood = np.array([[10, 22, 7, 1],
                      [4, 16, 16, 4],
                      [1, 7, 22, 10]]).T / 120
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=2, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)


def test_M33Shapes():
    NumberTest = 100
    for i in range(NumberTest):
        sides = 2 * np.random.rand(2, 2)
        M = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=3, sides=sides)
        np.testing.assert_array_equal(M.shape, (4, 4))


def test_M33SymetryMatrix():
    NumberTest = 100
    for i in range(NumberTest):
        sides = 2 * np.random.rand(2, 2)
        M = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=3, sides=sides)
        assert SUF.isSymetric(M) is True


def test_M33SymetrySides():
    NumberTest = 100
    for i in range(NumberTest):
        side0 = 2 * np.random.rand(2)
        sides = np.array([side0, side0])
        M = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=3, sides=sides)
        assert SUF.isSymetric(M, diagonal=2) is True


def test_M33SumAllValues():
    NumberTest = 100
    for i in range(NumberTest):
        sides = 2 * np.random.rand(2, 2)
        M = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=3, sides=sides)
        assert np.sum(M) == pytest.approx(1)


def test_M33Values():
    sides = np.array([[1, 1], [1, 1]])
    Mgood = np.array([[20, 129, 60, 1],
                      [129, 1188, 933, 60],
                      [60, 933, 1188, 129],
                      [1, 60, 129, 20]]) / 5040
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=3, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[1, 0], [1, 1]])
    Mgood = np.array([[90, 357, 180, 3],
                      [357, 2128, 1806, 119],
                      [180, 1806, 2376, 258],
                      [3, 119, 258, 40]]) / 10080
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=3, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[0, 0], [1, 1]])
    Mgood = np.array([[720, 441, 93, 6],
                      [441, 1071, 609, 84],
                      [93, 609, 563, 100],
                      [6, 84, 100, 20]]) / 5040
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=3, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[0, 0], [1, 0]])
    Mgood = np.array([[80, 49, 10, 1],
                      [49, 119, 63, 14],
                      [10, 63, 52, 15],
                      [1, 14, 15, 5]]) / 560
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=3, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[0, 0], [0, 0]])
    Mgood = np.array([[20, 10, 4, 1],
                      [10, 12, 9, 4],
                      [4, 9, 12, 10],
                      [1, 4, 10, 20]]) / 140
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=3, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[1, 0], [1, 0]])
    Mgood = np.array([[180, 714, 357, 9],
                      [714, 4256, 3493, 357],
                      [357, 3493, 4256, 714],
                      [9, 357, 714, 180]]) / 20160
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=3, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)

    sides = np.array([[0, 1], [0, 1]])
    Mgood = np.array([[20, 40, 9, 1],
                      [40, 108, 53, 9],
                      [9, 53, 108, 40],
                      [1, 9, 40, 20]]) / 560
    Mtest = SDI.SplineIntegralDouble.getIntegralBase(j=3, k=3, sides=sides)
    np.testing.assert_almost_equal(Mgood, Mtest)
