import pytest
import numpy as np
import femnurbs.SplineForceMatrix as SFM
import femnurbs.SplineUsefulFunctions as SUF


def test_ErrorBadParameter():
    # p = 1
    with pytest.raises(TypeError):
        SFM.getW()

    with pytest.raises(TypeError):
        SFM.getW(p=1)

    with pytest.raises(TypeError):
        SFM.getW(p=2)

    with pytest.raises(TypeError):
        SFM.getW(p=3)

    with pytest.raises(TypeError):
        SFM.getW(w=1)

    with pytest.raises(TypeError):
        SFM.getW(w=2)

    with pytest.raises(TypeError):
        SFM.getW(w=5)

    with pytest.raises(ValueError):
        SFM.getW(p=-1, w=1)

    with pytest.raises(ValueError):
        SFM.getW(p=-2, w=1)

    with pytest.raises(ValueError):
        SFM.getW(p=0, w=1)

    with pytest.raises(ValueError):
        SFM.getW(p=0, w=0)

    with pytest.raises(ValueError):
        SFM.getW(p=0, w=-1)

    with pytest.raises(NotImplementedError):
        SFM.getW(p=4, w=1)


def test_MatrixShape():
    Ntest = 100
    for p in (1, 2, 3):
        for w in range(1, 7):
            if p < 2:
                sides = None
            elif p == 2:
                sides = np.random.rand(2, 1)
            elif p == 3:
                sides = np.random.rand(2, 2)
            W = SFM.getW(p=p, w=w, sides=sides)
            np.testing.assert_array_equal(W.shape, (p + 1, w + 1))


def test_SumAllValues():
    Ntest = 100
    for p in (1, 2, 3):
        for w in range(1, 7):
            for zz in range(Ntest):
                if p < 2:
                    sides = None
                elif p == 2:
                    sides = np.random.rand(2, 1)
                elif p == 3:
                    sides = np.random.rand(2, 2)
                W = SFM.getW(p=p, w=w, sides=sides)
                assert np.sum(W) == pytest.approx(1)


def test_SymetryMatrix():
    Ntest = 100

    # p = 1
    for w in range(1, 7):
        W = SFM.getW(p=1, w=w)
        assert SUF.isSymetric(W, diagonal=2) == True

    # p = 2
    for w in range(1, 7):
        for zz in range(Ntest):
            lado = np.random.rand(1)
            sides = np.array([lado, lado])
            W = SFM.getW(p=2, w=w, sides=sides)
            assert SUF.isSymetric(W, diagonal=2) == True

    # p = 3
    for w in range(1, 7):
        for zz in range(Ntest):
            lado = np.random.rand(2)
            sides = np.array([lado, lado])
            W = SFM.getW(p=3, w=w, sides=sides)
            assert SUF.isSymetric(W, diagonal=2) == True


def test_W11():
    Wgood = [[2, 1],
             [1, 2]]
    Wgood = np.array(Wgood) / 6
    Wtest = SFM.getW(p=1, w=1)
    np.testing.assert_allclose(Wgood, Wtest)


def test_W12():
    Wgood = [[3, 2, 1],
             [1, 2, 3]]
    Wgood = np.array(Wgood) / 12
    Wtest = SFM.getW(p=1, w=2)
    np.testing.assert_allclose(Wgood, Wtest)


def test_W13():
    Wgood = [[4, 3, 2, 1],
             [1, 2, 3, 4]]
    Wgood = np.array(Wgood) / 20
    Wtest = SFM.getW(p=1, w=3)
    np.testing.assert_allclose(Wgood, Wtest)


def test_W14():
    Wgood = [[5, 4, 3, 2, 1],
             [1, 2, 3, 4, 5]]
    Wgood = np.array(Wgood) / 30
    Wtest = SFM.getW(p=1, w=4)
    np.testing.assert_allclose(Wgood, Wtest)


def test_W15():
    Wgood = [[6, 5, 4, 3, 2, 1],
             [1, 2, 3, 4, 5, 6]]
    Wgood = np.array(Wgood) / 42
    Wtest = SFM.getW(p=1, w=5)
    np.testing.assert_allclose(Wgood, Wtest)


def test_W16():
    Wgood = [[7, 6, 5, 4, 3, 2, 1],
             [1, 2, 3, 4, 5, 6, 7]]
    Wgood = np.array(Wgood) / 56
    Wtest = SFM.getW(p=1, w=6)
    np.testing.assert_allclose(Wgood, Wtest)


def test_W21():
    H = np.array([1, 1, 1])
    sides = SUF.transformHtoSides(H)
    Wgood = [[3, 1],
             [8, 8],
             [1, 3]]
    Wgood = np.array(Wgood) / 24
    Wtest = SFM.getW(p=2, w=1, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)

    H = np.array([0, 1, 0])
    sides = SUF.transformHtoSides(H)
    Wgood = [[3, 1],
             [2, 2],
             [1, 3]]
    Wgood = np.array(Wgood) / 12
    Wtest = SFM.getW(p=2, w=1, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)

    H = np.array([0, 1, 1])
    sides = SUF.transformHtoSides(H)
    Wgood = [[6, 2],
             [5, 7],
             [1, 3]]
    Wgood = np.array(Wgood) / 24
    Wtest = SFM.getW(p=2, w=1, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)


def test_W22():
    H = np.array([1, 1, 1])
    sides = SUF.transformHtoSides(H)
    Wgood = [[6, 3, 1],
             [13, 14, 13],
             [1, 3, 6]]
    Wgood = np.array(Wgood) / 60
    Wtest = SFM.getW(p=2, w=2, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)

    H = np.array([0, 1, 0])
    sides = SUF.transformHtoSides(H)
    Wgood = [[6, 3, 1],
             [3, 4, 3],
             [1, 3, 6]]
    Wgood = np.array(Wgood) / 30
    Wtest = SFM.getW(p=2, w=2, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)

    H = np.array([0, 1, 1])
    sides = SUF.transformHtoSides(H)
    Wgood = [[12, 6, 2],
             [7, 11, 12],
             [1, 3, 6]]
    Wgood = np.array(Wgood) / 60
    Wtest = SFM.getW(p=2, w=2, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)


def test_W23():
    H = np.array([1, 1, 1])
    sides = SUF.transformHtoSides(H)
    Wgood = [[10, 6, 3, 1],
             [19, 21, 21, 19],
             [1, 3, 6, 10]]
    Wgood = np.array(Wgood) / 120
    Wtest = SFM.getW(p=2, w=3, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)

    H = np.array([0, 1, 0])
    sides = SUF.transformHtoSides(H)
    Wgood = [[10, 6, 3, 1],
             [4, 6, 6, 4],
             [1, 3, 6, 10]]
    Wgood = np.array(Wgood) / 60
    Wtest = SFM.getW(p=2, w=3, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)

    H = np.array([0, 1, 1])
    sides = SUF.transformHtoSides(H)
    Wgood = [[20, 12, 6, 2],
             [9, 15, 18, 18],
             [1, 3, 6, 10]]
    Wgood = np.array(Wgood) / 120
    Wtest = SFM.getW(p=2, w=3, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)


def test_W24():
    H = np.array([1, 1, 1])
    sides = SUF.transformHtoSides(H)
    Wgood = [[15, 10, 6, 3, 1],
             [26, 29, 30, 29, 26],
             [1, 3, 6, 10, 15]]
    Wgood = np.array(Wgood) / 210
    Wtest = SFM.getW(p=2, w=4, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)

    H = np.array([0, 1, 0])
    sides = SUF.transformHtoSides(H)
    Wgood = [[15, 10, 6, 3, 1],
             [5, 8, 9, 8, 5],
             [1, 3, 6, 10, 15]]
    Wgood = np.array(Wgood) / 105
    Wtest = SFM.getW(p=2, w=4, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)

    H = np.array([0, 1, 1])
    sides = SUF.transformHtoSides(H)
    Wgood = [[30, 20, 12, 6, 2],
             [11, 19, 24, 26, 25],
             [1, 3, 6, 10, 15]]
    Wgood = np.array(Wgood) / 210
    Wtest = SFM.getW(p=2, w=4, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)


def test_W25():
    H = np.array([1, 1, 1])
    sides = SUF.transformHtoSides(H)
    Wgood = [[21, 15, 10, 6, 3, 1],
             [34, 38, 40, 40, 38, 34],
             [1, 3, 6, 10, 15, 21]]
    Wgood = np.array(Wgood) / 336
    Wtest = SFM.getW(p=2, w=5, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)

    H = np.array([0, 1, 0])
    sides = SUF.transformHtoSides(H)
    Wgood = [[21, 15, 10, 6, 3, 1],
             [6, 10, 12, 12, 10, 6],
             [1, 3, 6, 10, 15, 21]]
    Wgood = np.array(Wgood) / 168
    Wtest = SFM.getW(p=2, w=5, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)

    H = np.array([0, 1, 1])
    sides = SUF.transformHtoSides(H)
    Wgood = [[42, 30, 20, 12, 6, 2],
             [13, 23, 30, 34, 35, 33],
             [1, 3, 6, 10, 15, 21]]
    Wgood = np.array(Wgood) / 336
    Wtest = SFM.getW(p=2, w=5, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)


def test_W26():
    H = np.array([1, 1, 1])
    sides = SUF.transformHtoSides(H)
    Wgood = [[28, 21, 15, 10, 6, 3, 1],
             [43, 48, 51, 52, 51, 48, 43],
             [1, 3, 6, 10, 15, 21, 28]]
    Wgood = np.array(Wgood) / 504
    Wtest = SFM.getW(p=2, w=6, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)

    H = np.array([0, 1, 0])
    sides = SUF.transformHtoSides(H)
    Wgood = [[28, 21, 15, 10, 6, 3, 1],
             [7, 12, 15, 16, 15, 12, 7],
             [1, 3, 6, 10, 15, 21, 28]]
    Wgood = np.array(Wgood) / 252
    Wtest = SFM.getW(p=2, w=6, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)

    H = np.array([0, 1, 1])
    sides = SUF.transformHtoSides(H)
    Wgood = [[56, 42, 30, 20, 12, 6, 2],
             [15, 27, 36, 42, 45, 45, 42],
             [1, 3, 6, 10, 15, 21, 28]]
    Wgood = np.array(Wgood) / 504
    Wtest = SFM.getW(p=2, w=6, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)


def test_W31():
    H = np.array([1, 1, 1, 1, 1])
    sides = SUF.transformHtoSides(H)
    Wgood = [[4, 1],
             [33, 22],
             [22, 33],
             [1, 4]]
    Wgood = np.array(Wgood) / 120
    Wtest = SFM.getW(p=3, w=1, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)

    H = np.array([0, 1, 1, 1, 0])
    sides = SUF.transformHtoSides(H)
    Wgood = [[12, 3],
             [62, 43],
             [43, 62],
             [3, 12]]
    Wgood = np.array(Wgood) / 240
    Wtest = SFM.getW(p=3, w=1, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)

    H = np.array([0, 0, 1, 0, 0])
    sides = SUF.transformHtoSides(H)
    Wgood = [[4, 1],
             [3, 2],
             [2, 3],
             [1, 4]]
    Wgood = np.array(Wgood) / 20
    Wtest = SFM.getW(p=3, w=1, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)

    H = np.array([1, 0, 1, 0, 0])
    sides = SUF.transformHtoSides(H)
    Wgood = [[4, 1],
             [10, 5],
             [4, 6],
             [2, 8]]
    Wgood = np.array(Wgood) / 40
    Wtest = SFM.getW(p=3, w=1, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)


def test_W32():
    H = np.array([1, 1, 1, 1, 1])
    sides = SUF.transformHtoSides(H)
    Wgood = [[10, 4, 1],
             [71, 56, 38],
             [38, 56, 71],
             [1, 4, 10]]
    Wgood = np.array(Wgood) / 360
    Wtest = SFM.getW(p=3, w=2, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)

    H = np.array([0, 1, 1, 1, 0])
    sides = SUF.transformHtoSides(H)
    Wgood = [[10, 4, 1],
             [44, 36, 25],
             [25, 36, 44],
             [1, 4, 10]]
    Wgood = np.array(Wgood) / 240
    Wtest = SFM.getW(p=3, w=2, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)

    H = np.array([0, 0, 1, 0, 0])
    sides = SUF.transformHtoSides(H)
    Wgood = [[10, 4, 1],
             [6, 6, 3],
             [3, 6, 6],
             [1, 4, 10]]
    Wgood = np.array(Wgood) / 60
    Wtest = SFM.getW(p=3, w=2, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)

    H = np.array([1, 0, 1, 0, 0])
    sides = SUF.transformHtoSides(H)
    Wgood = [[10, 4, 1],
             [22, 16, 7],
             [6, 12, 12],
             [2, 8, 20]]
    Wgood = np.array(Wgood) / 120
    Wtest = SFM.getW(p=3, w=2, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)


def test_W33():
    H = np.array([1, 1, 1, 1, 1])
    sides = SUF.transformHtoSides(H)
    Wgood = [[20, 10, 4, 1],
             [129, 110, 86, 60],
             [60, 86, 110, 129],
             [1, 4, 10, 20]]
    Wgood = np.array(Wgood) / 840
    Wtest = SFM.getW(p=3, w=3, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)

    H = np.array([0, 1, 1, 1, 0])
    sides = SUF.transformHtoSides(H)
    Wgood = [[60, 30, 12, 3],
             [238, 210, 168, 119],
             [119, 168, 210, 238],
             [3, 12, 30, 60]]
    Wgood = np.array(Wgood) / 1680
    Wtest = SFM.getW(p=3, w=3, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)

    H = np.array([0, 0, 1, 0, 0])
    sides = SUF.transformHtoSides(H)
    Wgood = [[20, 10, 4, 1],
             [10, 12, 9, 4],
             [4, 9, 12, 10],
             [1, 4, 10, 20]]
    Wgood = np.array(Wgood) / 140
    Wtest = SFM.getW(p=3, w=3, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)

    H = np.array([1, 0, 1, 0, 0])
    sides = SUF.transformHtoSides(H)
    Wgood = [[20, 10, 4, 1],
             [40, 34, 22, 9],
             [8, 18, 24, 20],
             [2, 8, 20, 40]]
    Wgood = np.array(Wgood) / 280
    Wtest = SFM.getW(p=3, w=3, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)


def test_W34():
    H = np.array([1, 1, 1, 1, 1])
    sides = SUF.transformHtoSides(H)
    Wgood = [[35, 20, 10, 4, 1],
             [211, 188, 158, 124, 89],
             [89, 124, 158, 188, 211],
             [1, 4, 10, 20, 35]]
    Wgood = np.array(Wgood) / 1680
    Wtest = SFM.getW(p=3, w=4, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)

    H = np.array([0, 1, 1, 1, 0])
    sides = SUF.transformHtoSides(H)
    Wgood = [[105, 60, 30, 12, 3],
             [387, 356, 306, 244, 177],
             [177, 244, 306, 356, 387],
             [3, 12, 30, 60, 105]]
    Wgood = np.array(Wgood) / 3360
    Wtest = SFM.getW(p=3, w=4, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)

    H = np.array([0, 0, 1, 0, 0])
    sides = SUF.transformHtoSides(H)
    Wgood = [[35, 20, 10, 4, 1],
             [15, 20, 18, 12, 5],
             [5, 12, 18, 20, 15],
             [1, 4, 10, 20, 35]]
    Wgood = np.array(Wgood) / 280
    Wtest = SFM.getW(p=3, w=4, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)

    H = np.array([1, 0, 1, 0, 0])
    sides = SUF.transformHtoSides(H)
    Wgood = [[35, 20, 10, 4, 1],
             [65, 60, 46, 28, 11],
             [10, 24, 36, 40, 30],
             [2, 8, 20, 40, 70]]
    Wgood = np.array(Wgood) / 560
    Wtest = SFM.getW(p=3, w=4, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)


def test_W35():
    H = np.array([1, 1, 1, 1, 1])
    sides = SUF.transformHtoSides(H)
    Wgood = [[56, 35, 20, 10, 4, 1],
             [321, 294, 258, 216, 171, 126],
             [126, 171, 216, 258, 294, 321],
             [1, 4, 10, 20, 35, 56]]
    Wgood = np.array(Wgood) / 3024
    Wtest = SFM.getW(p=3, w=5, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)

    H = np.array([0, 1, 1, 1, 0])
    sides = SUF.transformHtoSides(H)
    Wgood = [[168, 105, 60, 30, 12, 3],
             [586, 553, 496, 422, 338, 251],
             [251, 338, 422, 496, 553, 586],
             [3, 12, 30, 60, 105, 168]]
    Wgood = np.array(Wgood) / 6048
    Wtest = SFM.getW(p=3, w=5, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)

    H = np.array([0, 0, 1, 0, 0])
    sides = SUF.transformHtoSides(H)
    Wgood = [[56, 35, 20, 10, 4, 1],
             [21, 30, 30, 24, 15, 6],
             [6, 15, 24, 30, 30, 21],
             [1, 4, 10, 20, 35, 56]]
    Wgood = np.array(Wgood) / 504
    Wtest = SFM.getW(p=3, w=5, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)

    H = np.array([1, 0, 1, 0, 0])
    sides = SUF.transformHtoSides(H)
    Wgood = [[56, 35, 20, 10, 4, 1],
             [98, 95, 80, 58, 34, 13],
             [12, 30, 48, 60, 60, 42],
             [2, 8, 20, 40, 70, 112]]
    Wgood = np.array(Wgood) / 1008
    Wtest = SFM.getW(p=3, w=5, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)


def test_W36():
    H = np.array([1, 1, 1, 1, 1])
    sides = SUF.transformHtoSides(H)
    Wgood = [[84, 56, 35, 20, 10, 4, 1],
             [463, 432, 390, 340, 285, 228, 172],
             [172, 228, 285, 340, 390, 432, 463],
             [1, 4, 10, 20, 35, 56, 84]]
    Wgood = np.array(Wgood) / 5040
    Wtest = SFM.getW(p=3, w=6, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)

    H = np.array([0, 1, 1, 1, 1])
    sides = SUF.transformHtoSides(H)
    Wgood = [[252, 168, 105, 60, 30, 12, 3],
             [842, 808, 745, 660, 560, 452, 343],
             [344, 456, 570, 680, 780, 864, 926],
             [2, 8, 20, 40, 70, 112, 168]]
    Wgood = np.array(Wgood) / 10080
    Wtest = SFM.getW(p=3, w=6, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)

    H = np.array([0, 1, 1, 1, 0])
    sides = SUF.transformHtoSides(H)
    Wgood = [[252, 168, 105, 60, 30, 12, 3],
             [842, 808, 745, 660, 560, 452, 343],
             [343, 452, 560, 660, 745, 808, 842],
             [3, 12, 30, 60, 105, 168, 252]]
    Wgood = np.array(Wgood) / 10080
    Wtest = SFM.getW(p=3, w=6, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)

    H = np.array([0, 0, 1, 0, 0])
    sides = SUF.transformHtoSides(H)
    Wgood = [[84, 56, 35, 20, 10, 4, 1],
             [28, 42, 45, 40, 30, 18, 7],
             [7, 18, 30, 40, 45, 42, 28],
             [1, 4, 10, 20, 35, 56, 84]]
    Wgood = np.array(Wgood) / 840
    Wtest = SFM.getW(p=3, w=6, sides=sides)
    np.testing.assert_allclose(Wgood, Wtest)


def test_WAllDomainShapes():
    for p in (1, 2, 3):
        for n in range(p + 1, 10):
            U = SUF.UUniform(p=p, n=n)
            for w in range(1, 7):
                for j in range(1, p + 1):
                    Wtest = SFM.getWAllDomain(U, j=j, w=w)
                    correct_shape = (n + j - p, w * (n - p) + 1)
                    np.testing.assert_array_equal(Wtest.shape, correct_shape)


def test_WAllDomainSumAllValues():
    for p in (1, 2, 3):
        for n in range(p + 1, 10):
            U = SUF.UUniform(p=p, n=n)
            for w in range(1, 7):
                for j in range(1, p + 1):
                    Wtest = SFM.getWAllDomain(U, j=j, w=w)
                    assert np.sum(Wtest) == pytest.approx(1)


def test_WAllDomainBezierP1():
    p = 1
    U = SUF.UBezier(p=p)
    Wtest = SFM.getWAllDomain(U, j=1, w=1)
    Wgood = [[2, 1],
             [1, 2]]
    Wgood = np.array(Wgood) / 6
    np.testing.assert_allclose(Wgood, Wtest)

    Wtest = SFM.getWAllDomain(U, j=1, w=2)
    Wgood = [[3, 2, 1],
             [1, 2, 3]]
    Wgood = np.array(Wgood) / 12

    Wtest = SFM.getWAllDomain(U, j=1, w=3)
    Wgood = [[4, 3, 2, 1],
             [1, 2, 3, 4]]
    Wgood = np.array(Wgood) / 20

    Wtest = SFM.getWAllDomain(U, j=1, w=4)
    Wgood = [[5, 4, 3, 2, 1],
             [1, 2, 3, 4, 5]]
    Wgood = np.array(Wgood) / 30
    np.testing.assert_allclose(Wgood, Wtest)


def test_WAllDomainBezierP2():
    p = 2
    U = SUF.UBezier(p=p)

    Wtest = SFM.getWAllDomain(U, j=2, w=1)
    Wgood = [[3, 1],
             [2, 2],
             [1, 3]]
    Wgood = np.array(Wgood) / 12
    np.testing.assert_allclose(Wgood, Wtest)

    Wtest = SFM.getWAllDomain(U, j=2, w=2)
    Wgood = [[6, 3, 1],
             [3, 4, 3],
             [1, 3, 6]]
    Wgood = np.array(Wgood) / 30

    Wtest = SFM.getWAllDomain(U, j=2, w=3)

    Wgood = [[10, 6, 3, 1],
             [4, 6, 6, 4],
             [1, 3, 6, 10]]
    Wgood = np.array(Wgood) / 60

    Wtest = SFM.getWAllDomain(U, j=2, w=4)

    Wgood = [[15, 10, 6, 3, 1],
             [5, 8, 9, 8, 5],
             [1, 3, 6, 10, 15]]
    Wgood = np.array(Wgood) / 105
    np.testing.assert_allclose(Wgood, Wtest)


def test_WAllDomainBezierP3():
    p = 3
    U = SUF.UBezier(p=p)

    Wtest = SFM.getWAllDomain(U, j=3, w=1)
    Wgood = [[12, 3],
             [9, 6],
             [6, 9],
             [3, 12]]
    Wgood = np.array(Wgood) / 60
    np.testing.assert_allclose(Wgood, Wtest)

    Wtest = SFM.getWAllDomain(U, j=3, w=2)
    Wgood = [[10, 4, 1],
             [6, 6, 3],
             [3, 6, 6],
             [1, 4, 10]]
    Wgood = np.array(Wgood) / 60

    Wtest = SFM.getWAllDomain(U, j=3, w=3)
    Wgood = [[20, 10, 4, 1],
             [10, 12, 9, 4],
             [4, 9, 12, 10],
             [1, 4, 10, 20]]
    Wgood = np.array(Wgood) / 140

    Wtest = SFM.getWAllDomain(U, j=3, w=4)
    fator = 3 * 4 * 5 * 7 * 2 / 3
    Wgood = [[35, 20, 10, 4, 1],
             [15, 20, 18, 12, 5],
             [5, 12, 18, 20, 15],
             [1, 4, 10, 20, 35]]
    Wgood = np.array(Wgood) / 280
    np.testing.assert_allclose(Wgood, Wtest)


def test_WAllDomainUniformP1N3():
    p, n = 1, 3
    U = SUF.UUniform(p=p, n=n)

    Wtest = SFM.getWAllDomain(U, j=1, w=1)
    Wgood = [[2, 1, 0],
             [1, 4, 1],
             [0, 1, 2]]
    Wgood = np.array(Wgood) / 12
    np.testing.assert_allclose(Wgood, Wtest)

    Wtest = SFM.getWAllDomain(U, j=1, w=2)
    Wgood = [[3, 2, 1, 0, 0],
             [1, 2, 6, 2, 1],
             [0, 0, 1, 2, 3]]
    Wgood = np.array(Wgood) / 24
    np.testing.assert_allclose(Wgood, Wtest)

    Wtest = SFM.getWAllDomain(U, j=1, w=3)
    Wgood = [[4, 3, 2, 1, 0, 0, 0],
             [1, 2, 3, 8, 3, 2, 1],
             [0, 0, 0, 1, 2, 3, 4]]
    Wgood = np.array(Wgood) / 40
    np.testing.assert_allclose(Wgood, Wtest)

    Wtest = SFM.getWAllDomain(U, j=1, w=4)
    Wgood = [[5, 4, 3, 2, 1, 0, 0, 0, 0],
             [1, 2, 3, 4, 10, 4, 3, 2, 1],
             [0, 0, 0, 0, 1, 2, 3, 4, 5]]
    Wgood = np.array(Wgood) / 60
    np.testing.assert_allclose(Wgood, Wtest)


def test_WAllDomainUniformP1N4():
    p, n = 1, 4
    U = SUF.UUniform(p=p, n=n)

    Wtest = SFM.getWAllDomain(U, j=1, w=1)
    Wgood = [[2, 1, 0, 0],
             [1, 4, 1, 0],
             [0, 1, 4, 1],
             [0, 0, 1, 2]]
    Wgood = np.array(Wgood) / 18
    np.testing.assert_allclose(Wgood, Wtest)

    Wtest = SFM.getWAllDomain(U, j=1, w=2)

    Wgood = [[3, 2, 1, 0, 0, 0, 0],
             [1, 2, 6, 2, 1, 0, 0],
             [0, 0, 1, 2, 6, 2, 1],
             [0, 0, 0, 0, 1, 2, 3]]
    Wgood = np.array(Wgood) / 36
    np.testing.assert_allclose(Wgood, Wtest)

    Wtest = SFM.getWAllDomain(U, j=1, w=3)
    Wgood = [[4, 3, 2, 1, 0, 0, 0, 0, 0, 0],
             [1, 2, 3, 8, 3, 2, 1, 0, 0, 0],
             [0, 0, 0, 1, 2, 3, 8, 3, 2, 1],
             [0, 0, 0, 0, 0, 0, 1, 2, 3, 4]]
    Wgood = np.array(Wgood) / 60
    np.testing.assert_allclose(Wgood, Wtest)

    Wtest = SFM.getWAllDomain(U, j=1, w=4)
    Wgood = [[5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 2, 3, 4, 10, 4, 3, 2, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 2, 3, 4, 10, 4, 3, 2, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5]]
    Wgood = np.array(Wgood) / 90
    np.testing.assert_allclose(Wgood, Wtest)



def test_SumAllComponentsForceAllDomain_UUniform_FunctionConstant():
    def f(u):
        return 0
    for p in (1, 2, 3):
        for n in range(p + 1, 10):
            U = SUF.UUniform(p=p, n=n)
            for w in range(1, 7):
                Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
                Fgood = np.zeros(n)
                np.testing.assert_array_equal(Ftest, Fgood)
                assert np.sum(Ftest) == pytest.approx(0)

    def f(u):
        return 1
    for p in (1, 2, 3):
        for n in range(p + 1, 10):
            U = SUF.UUniform(p=p, n=n)
            for w in range(1, 7):
                Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
                assert np.sum(Ftest) == pytest.approx(1)

def test_SumAllComponentsForceAllDomain_UUniform_FunctionLinear():
    def f(u):
        return u
    for p in (1, 2, 3):
        for n in range(p + 1, 10):
            U = SUF.UUniform(p=p, n=n)
            for w in range(1, 7):
                Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
                assert np.sum(Ftest) == pytest.approx(1/2)

    def f(u):
        return 1-u
    for p in (1, 2, 3):
        for n in range(p + 1, 10):
            U = SUF.UUniform(p=p, n=n)
            for w in range(1, 7):
                Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
                assert np.sum(Ftest) == pytest.approx(1/2)
    def f(u):
        return 2*u-1
    for p in (1, 2, 3):
        for n in range(p + 1, 10):
            U = SUF.UUniform(p=p, n=n)
            for w in range(1, 7):
                Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
                assert np.sum(Ftest) == pytest.approx(0)

@pytest.mark.skip(reason="For force function degree 2 doesn't work yet")
def test_SumAllComponentsForceAllDomain_UUniform_FunctionQuadratic():
    def f(u):
        return u**2
    for p in (1, 2, 3):
        for n in range(p + 1, 10):
            U = SUF.UUniform(p=p, n=n)
            for w in range(2, 7):
                Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
                assert np.sum(Ftest) == pytest.approx(1/3)

    def f(u):
        return 2*u*(1-u)
    for p in (1, 2, 3):
        for n in range(p + 1, 10):
            U = SUF.UUniform(p=p, n=n)
            for w in range(2, 7):
                Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
                assert np.sum(Ftest) == pytest.approx(1/3)

    def f(u):
        return (1-u)**2
    for p in (1, 2, 3):
        for n in range(p + 1, 10):
            U = SUF.UUniform(p=p, n=n)
            for w in range(2, 7):
                Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
                assert np.sum(Ftest) == pytest.approx(1/3)


def test_SumAllComponentsForceAllDomain_URandom_FunctionConstant():
    def f(u):
        return 0
    for p in (1, 2, 3):
        for n in range(p + 1, 10):
            U = SUF.URandom(p=p, n=n)
            for w in range(1, 7):
                Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
                Fgood = np.zeros(n)
                np.testing.assert_array_equal(Ftest, Fgood)
                assert np.sum(Ftest) == pytest.approx(0)

    def f(u):
        return 1
    for p in (1, 2, 3):
        for n in range(p + 1, 10):
            U = SUF.URandom(p=p, n=n)
            for w in range(1, 7):
                Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
                assert np.sum(Ftest) == pytest.approx(1)

def test_SumAllComponentsForceAllDomain_URandom_FunctionLinear():
    def f(u):
        return u
    for p in (1, 2, 3):
        for n in range(p + 1, 10):
            U = SUF.URandom(p=p, n=n)
            for w in range(1, 7):
                Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
                assert np.sum(Ftest) == pytest.approx(1/2)

    def f(u):
        return 1-u
    for p in (1, 2, 3):
        for n in range(p + 1, 10):
            U = SUF.URandom(p=p, n=n)
            for w in range(1, 7):
                Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
                assert np.sum(Ftest) == pytest.approx(1/2)

    def f(u):
        return 2*u-1
    for p in (1, 2, 3):
        for n in range(p + 1, 10):
            U = SUF.URandom(p=p, n=n)
            for w in range(1, 7):
                Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
                assert np.sum(Ftest) == pytest.approx(0)

@pytest.mark.skip(reason="For force function degree 2 doesn't work yet")
def test_SumAllComponentsForceAllDomain_URandom():
    def f(u):
        return u**2
    for p in (1, 2, 3):
        for n in range(p + 1, 10):
            U = SUF.URandom(p=p, n=n)
            for w in range(2, 7):
                Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
                assert np.sum(Ftest) == pytest.approx(1/3)

    def f(u):
        return 2*u*(1-u)
    for p in (1, 2, 3):
        for n in range(p + 1, 10):
            U = SUF.URandom(p=p, n=n)
            for w in range(2, 7):
                Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
                assert np.sum(Ftest) == pytest.approx(1/3)

    def f(u):
        return (1-u)**2
    for p in (1, 2, 3):
        for n in range(p + 1, 10):
            U = SUF.URandom(p=p, n=n)
            for w in range(2, 7):
                Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
                assert np.sum(Ftest) == pytest.approx(1/3)


def test_computeForceAllDomainForceConstant():
    def f(u):
        return 1

    if True:
        p, n = 1, 2
        U = SUF.UBezier(p=p)
        Fgood = np.ones(n) / n
        for w in (1, 2, 3, 4, 5, 6):
            Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
            np.testing.assert_allclose(Ftest, Fgood)

        p, n = 1, 3
        U = SUF.UUniform(p=p, n=n)
        Fgood = np.array([1, 2, 1]) / 4
        for w in (1, 2, 3, 4, 5, 6):
            Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
            np.testing.assert_allclose(Ftest, Fgood)

        p, n = 1, 4
        U = SUF.UUniform(p=p, n=n)
        Fgood = np.array([1, 2, 2, 1]) / 6
        for w in (1, 2, 3, 4, 5, 6):
            Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
            np.testing.assert_allclose(Ftest, Fgood)

        p, n = 1, 5
        U = SUF.UUniform(p=p, n=n)
        Fgood = np.array([1, 2, 2, 2, 1]) / 8
        for w in (1, 2, 3, 4, 5, 6):
            Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
            np.testing.assert_allclose(Ftest, Fgood)

    if True:
        p, n = 2, 3
        U = SUF.UBezier(p=p)
        Fgood = np.ones(n) / (n)
        for w in (1, 2, 3, 4, 5, 6):
            Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
            np.testing.assert_allclose(Ftest, Fgood)

        p, n = 2, 4
        U = SUF.UUniform(p=p, n=n)
        Fgood = np.array([1, 2, 2, 1]) / 6
        for w in (1, 2, 3, 4, 5, 6):
            Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
            np.testing.assert_allclose(Ftest, Fgood)

        p, n = 2, 5
        U = SUF.UUniform(p=p, n=n)
        Fgood = np.array([1, 2, 3, 2, 1]) / 9
        for w in (1, 2, 3, 4, 5, 6):
            Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
            np.testing.assert_allclose(Ftest, Fgood)

        p, n = 2, 6
        U = SUF.UUniform(p=p, n=n)
        Fgood = np.array([1, 2, 3, 3, 2, 1]) / 12
        for w in (1, 2, 3, 4, 5, 6):
            Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
            np.testing.assert_allclose(Ftest, Fgood)

    if True:
        p, n = 3, 4
        U = SUF.UBezier(p=p)
        Fgood = np.ones(n) / (n)
        for w in (1, 2, 3, 4, 5, 6):
            Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
            np.testing.assert_allclose(Ftest, Fgood)

        p, n = 3, 5
        U = SUF.UUniform(p=p, n=n)
        Fgood = np.array([1, 2, 2, 2, 1]) / 8
        for w in (1, 2, 3, 4, 5, 6):
            Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
            np.testing.assert_allclose(Ftest, Fgood)

        p, n = 3, 6
        U = SUF.UUniform(p=p, n=n)
        Fgood = np.array([1, 2, 3, 3, 2, 1]) / 12
        for w in (1, 2, 3, 4, 5, 6):
            Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
            np.testing.assert_allclose(Ftest, Fgood)

        p, n = 3, 7
        U = SUF.UUniform(p=p, n=n)
        Fgood = np.array([1, 2, 3, 4, 3, 2, 1]) / 16
        for w in (1, 2, 3, 4, 5, 6):
            Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
            np.testing.assert_allclose(Ftest, Fgood)



def test_computeForceAllDomainForceLinear():
    def f(u):
        return u

    if True:
        p, n = 1, 2
        U = SUF.UBezier(p=p)
        Fgood = np.array([1, 2]) / 6
        for w in (1, 2, 3, 4, 5, 6):
            Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
            np.testing.assert_allclose(Ftest, Fgood)

        p, n = 1, 3
        U = SUF.UUniform(p=p, n=n)
        Fgood = np.array([1, 6, 5]) / 24
        for w in (1, 2, 3, 4, 5, 6):
            Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
            np.testing.assert_allclose(Ftest, Fgood)

        p, n = 1, 4
        U = SUF.UUniform(p=p, n=n)
        Fgood = np.array([1, 6, 12, 8]) / 54
        for w in (1, 2, 3, 4, 5, 6):
            Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
            np.testing.assert_allclose(Ftest, Fgood)

        p, n = 1, 5
        U = SUF.UUniform(p=p, n=n)
        Fgood = np.array([1, 6, 12, 18, 11]) / 96
        for w in (1, 2, 3, 4, 5, 6):
            Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
            np.testing.assert_allclose(Ftest, Fgood)

    if True:
        p, n = 2, 3
        U = SUF.UBezier(p=p)
        Fgood = np.array([1, 2, 3]) / 12
        for w in (1, 2, 3, 4, 5, 6):
            Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
            np.testing.assert_allclose(Ftest, Fgood)

        p, n = 2, 4
        U = SUF.UUniform(p=p, n=n)
        Fgood = np.array([1, 6, 10, 7]) / 48
        for w in (1, 2, 3, 4, 5, 6):
            Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
            np.testing.assert_allclose(Ftest, Fgood)

        p, n = 2, 5
        U = SUF.UUniform(p=p, n=n)
        Fgood = np.array([1, 6, 18, 18, 11]) / 108
        for w in (1, 2, 3, 4, 5, 6):
            Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
            np.testing.assert_allclose(Ftest, Fgood)

        p, n = 2, 6
        U = SUF.UUniform(p=p, n=n)
        Fgood = np.array([1, 6, 18, 30, 26, 15]) / 192
        for w in (1, 2, 3, 4, 5, 6):
            Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
            np.testing.assert_allclose(Ftest, Fgood)

    if True:
        p, n = 3, 4
        U = SUF.UBezier(p=p)
        Fgood = np.array([1, 2, 3, 4]) / 20
        for w in (1, 2, 3, 4, 5, 6):
            Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
            np.testing.assert_allclose(Ftest, Fgood)

        p, n = 3, 5
        U = SUF.UUniform(p=p, n=n)
        Fgood = np.array([1, 6, 10, 14, 9]) / 80
        for w in (1, 2, 3, 4, 5, 6):
            Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
            np.testing.assert_allclose(Ftest, Fgood)

        p, n = 3, 6
        U = SUF.UUniform(p=p, n=n)
        Fgood = np.array([1, 6, 18, 27, 24, 14]) / 180
        for w in (1, 2, 3, 4, 5, 6):
            Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
            np.testing.assert_allclose(Ftest, Fgood)

        p, n = 3, 7
        U = SUF.UUniform(p=p, n=n)
        Fgood = np.array([1, 6, 18, 40, 42, 34, 19]) / 320
        for w in (1, 2, 3, 4, 5, 6):
            Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
            np.testing.assert_allclose(Ftest, Fgood)


    # def f(u):
    #     return 2*u*(1-u)
    # for p in (1, 2, 3):
    #     for n in range(p+1, 10):
    #         U = SUF.UUniform(p=p, n=n)
    #         for w in (1, 2, 3, 4, 5, 6):
    #             for j in range(p + 1, 3):
    #                 Ftest = SFM.computeForceResultant(U, j=p, w=w, f=f)
    #                 Fgood = np.zeros(n) / (n - p + 1)
    #                 np.testing.assert_allclose(Ftest, Fgood)


if __name__ == "__main__":
    test_WAllDomainShapes()
    test_WAllDomainSumAllValues()
    test_WAllDomainBezierP1()
    test_WAllDomainBezierP2()
    test_WAllDomainBezierP3()
    test_WAllDomainUniformP1N3()
    test_WAllDomainUniformP1N4()
    test_SumAllComponentsForceAllDomainUUniformP1()
    test_SumAllComponentsForceAllDomainUUniformP2()
    test_SumAllComponentsForceAllDomainURandomP1()
    test_SumAllComponentsForceAllDomainURandomP2()
    # test_computeForceAllDomainForceNul()
    # test_computeForceAllDomainForceConstant()
    # test_computeForceAllDomainForceLinear()
