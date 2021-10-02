import pytest
import numpy as np
import femnurbs.SplineBaseFunction as SBF
import femnurbs.SplineUsefulFunctions as SUF


def test_CreationClass():
    SBF.SplineBaseFunction(SUF.UBezier(p=1))
    SBF.SplineBaseFunction(SUF.UBezier(p=2))
    SBF.SplineBaseFunction(SUF.UBezier(p=3))
    SBF.SplineBaseFunction(SUF.UUniform(p=3, n=5))

    with pytest.raises(TypeError):
        SBF.SplineBaseFunction(-1)
    with pytest.raises(TypeError):
        SBF.SplineBaseFunction({1: 1})

    with pytest.raises(ValueError):
        SBF.SplineBaseFunction([0, 0, 0, 1, 1])
    with pytest.raises(ValueError):
        SBF.SplineBaseFunction([0, 0, 1, 1, 1, ])
    with pytest.raises(ValueError):
        SBF.SplineBaseFunction([0, 0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError):
        SBF.SplineBaseFunction([0, 0, 0, 1, 1, 1, 1])
    with pytest.raises(ValueError):
        SBF.SplineBaseFunction([-1, -1, 1, 1])
    with pytest.raises(ValueError):
        SBF.SplineBaseFunction([0, 0, 2, 2])


def test_ValuesOfP():
    N = SBF.SplineBaseFunction(SUF.UBezier(p=1))
    assert N.p == 1
    N = SBF.SplineBaseFunction(SUF.UBezier(p=2))
    assert N.p == 2
    N = SBF.SplineBaseFunction(SUF.UBezier(p=3))
    assert N.p == 3

    N = SBF.SplineBaseFunction(SUF.UUniform(p=1, n=3))
    assert N.p == 1
    N = SBF.SplineBaseFunction([0, 0, 0.2, 0.6, 1, 1])
    assert N.p == 1
    N = SBF.SplineBaseFunction(SUF.UUniform(p=2, n=4))
    assert N.p == 2
    N = SBF.SplineBaseFunction([0, 0, 0, 0.2, 0.6, 1, 1, 1])
    assert N.p == 2
    N = SBF.SplineBaseFunction(SUF.UUniform(p=3, n=5))
    assert N.p == 3
    N = SBF.SplineBaseFunction([0, 0, 0, 0, 0.2, 0.6, 1, 1, 1, 1])
    assert N.p == 3


def test_ValuesOfN():
    N = SBF.SplineBaseFunction(SUF.UBezier(p=1))
    assert N.n == 2
    N = SBF.SplineBaseFunction(SUF.UBezier(p=2))
    assert N.n == 3
    N = SBF.SplineBaseFunction(SUF.UBezier(p=3))
    assert N.n == 4
    N = SBF.SplineBaseFunction(SUF.UBezier(p=4))
    assert N.n == 5

    N = SBF.SplineBaseFunction(SUF.UUniform(p=1, n=3))
    assert N.n == 3
    N = SBF.SplineBaseFunction([0, 0, 0.2, 0.6, 1, 1])
    assert N.n == 4
    N = SBF.SplineBaseFunction(SUF.UUniform(p=2, n=4))
    assert N.n == 4
    N = SBF.SplineBaseFunction([0, 0, 0, 0.2, 0.6, 1, 1, 1])
    assert N.n == 5
    N = SBF.SplineBaseFunction(SUF.UUniform(p=3, n=5))
    assert N.n == 5
    N = SBF.SplineBaseFunction([0, 0, 0, 0, 0.2, 0.6, 1, 1, 1, 1])
    assert N.n == 6


def test_CallableFunctions():
    U = [0, 0, 0.3, 0.7, 1, 1]  # p = 1 and n = 4
    N = SBF.SplineBaseFunction(U)

    assert callable(N) is True

    assert callable(N[0]) is True
    assert callable(N[1]) is True
    assert callable(N[2]) is True
    assert callable(N[3]) is True
    with pytest.raises(IndexError):
        N[4]
    with pytest.raises(IndexError):
        N[5]

    assert callable(N[:, 0]) is True
    assert callable(N[:, 1]) is True
    with pytest.raises(IndexError):
        N[:, 2]

    assert callable(N[0, 0]) is True
    assert callable(N[1, 0]) is True
    assert callable(N[2, 0]) is True
    assert callable(N[3, 0]) is True
    assert callable(N[0, 1]) is True
    assert callable(N[1, 1]) is True
    assert callable(N[2, 1]) is True
    assert callable(N[3, 1]) is True
    with pytest.raises(IndexError):
        N[0, 2]
    with pytest.raises(IndexError):
        N[1, 2]
    with pytest.raises(IndexError):
        N[2, 2]
    with pytest.raises(IndexError):
        N[3, 2]
    with pytest.raises(IndexError):
        N[4, 0]
    with pytest.raises(IndexError):
        N[4, 1]


def test_EvaluationWithVectorsShapes():
    U = SUF.UBezier(p=1)  # p = 1, n = 2
    # Ueval = [0, 0.2, 0.3, 0.5, 0.9, 1]
    Ueval = np.random.rand(6)

    N = SBF.SplineBaseFunction(U)
    assert isinstance(N(Ueval), np.ndarray)
    np.testing.assert_array_equal(N(Ueval).shape, (2, 6))
    assert isinstance(N[0](Ueval), np.ndarray)
    np.testing.assert_array_equal(N[0](Ueval).shape, (6))
    assert isinstance(N[1](Ueval), np.ndarray)
    np.testing.assert_array_equal(N[1](Ueval).shape, (6))
    assert isinstance(N[:, 0](Ueval), np.ndarray)
    np.testing.assert_array_equal(N[:, 0](Ueval).shape, (2, 6))
    assert isinstance(N[:, 1](Ueval), np.ndarray)
    np.testing.assert_array_equal(N[:, 1](Ueval).shape, (2, 6))


def test_EvaluationWithVectorsValues():
    if True:
        U = SUF.UBezier(p=1)
        Ueval = [0, 0.2, 0.3, 0.5, 0.9, 1]
        N = SBF.SplineBaseFunction(U)

        # j = 0
        Mtest = [[0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1]]
        np.testing.assert_almost_equal(N[:, 0](Ueval), Mtest)
        V = N[0, 0](Ueval)
        np.testing.assert_almost_equal(V, Mtest[0])
        V = N[1, 0](Ueval)
        np.testing.assert_almost_equal(V, Mtest[1])

        # j = 1
        Mtest = [[1, 0.8, 0.7, 0.5, 0.1, 0],
                 [0, 0.2, 0.3, 0.5, 0.9, 1]]
        np.testing.assert_almost_equal(N(Ueval), Mtest)
        V = N[0](Ueval)
        np.testing.assert_almost_equal(V, Mtest[0])
        V = N[1](Ueval)
        np.testing.assert_almost_equal(V, Mtest[1])

    if True:
        U = SUF.UBezier(p=2)
        Ueval = [0, 0.2, 0.3, 0.5, 0.9, 1]
        N = SBF.SplineBaseFunction(U)

        # j = 0
        Mtest = [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1]]
        np.testing.assert_almost_equal(N[:, 0](Ueval), Mtest)
        V = N[0, 0](Ueval)
        np.testing.assert_almost_equal(V, Mtest[0])
        V = N[1, 0](Ueval)
        np.testing.assert_almost_equal(V, Mtest[1])
        V = N[2, 0](Ueval)
        np.testing.assert_almost_equal(V, Mtest[2])

        # j = 1
        Mtest = [[0, 0, 0, 0, 0, 0],
                 [1, 0.8, 0.7, 0.5, 0.1, 0],
                 [0, 0.2, 0.3, 0.5, 0.9, 1]]
        np.testing.assert_almost_equal(N[:, 1](Ueval), Mtest)
        V = N[0, 1](Ueval)
        np.testing.assert_almost_equal(V, Mtest[0])
        V = N[1, 1](Ueval)
        np.testing.assert_almost_equal(V, Mtest[1])
        V = N[2, 1](Ueval)
        np.testing.assert_almost_equal(V, Mtest[2])

        # j = 2
        Mtest = [[1, 0.64, 0.49, 0.25, 0.01, 0],
                 [0, 0.32, 0.42, 0.5, 0.18, 0],
                 [0, 0.04, 0.09, 0.25, 0.81, 1]]
        np.testing.assert_almost_equal(N(Ueval), Mtest)
        V = N[0](Ueval)
        np.testing.assert_almost_equal(V, Mtest[0])
        V = N[1](Ueval)
        np.testing.assert_almost_equal(V, Mtest[1])
        V = N[2](Ueval)
        np.testing.assert_almost_equal(V, Mtest[2])


def test_BezierOrder1():
    n, p, D = 2, 1, 6
    U = SUF.UBezier(p=p)
    N = SBF.SplineBaseFunction(U)
    Ueval = np.linspace(0, 1, D)

    Mtest = N[:, 0](Ueval)
    Mgood = ((0, 0, 0, 0, 0, 0),
             (1, 1, 1, 1, 1, 1))
    np.testing.assert_array_equal(Mtest.shape, (n, D))
    np.testing.assert_almost_equal(Mtest, Mgood)
    for i in range(n):
        for j, u in enumerate(Ueval):
            assert N[i, 0](u) == pytest.approx(Mgood[i][j])

    Mtest = N[:, 1](Ueval)
    Mgood = ((1, 0.8, 0.6, 0.4, 0.2, 0),
             (0, 0.2, 0.4, 0.6, 0.8, 1))
    np.testing.assert_array_equal(Mtest.shape, (n, D))
    np.testing.assert_almost_equal(Mtest, Mgood)
    for i in range(n):
        for j, u in enumerate(Ueval):
            assert N[i, 1](u) == pytest.approx(Mgood[i][j])


def test_BezierOrder2():
    p, n, D = 2, 3, 6
    U = SUF.UBezier(p=p)
    N = SBF.SplineBaseFunction(U)
    Ueval = np.linspace(0, 1, D)

    # j = 0
    Mtest = N[:, 0](Ueval)
    Mgood = ((0, 0, 0, 0, 0, 0),
             (0, 0, 0, 0, 0, 0),
             (1, 1, 1, 1, 1, 1))
    np.testing.assert_array_equal(Mtest.shape, (n, D))
    np.testing.assert_almost_equal(Mtest, Mgood)
    for i in range(n):
        for j, u in enumerate(Ueval):
            assert N[i, 0](u) == pytest.approx(Mgood[i][j])

    # j = 1
    Mtest = N[:, 1](Ueval)
    Mgood = ((0, 0, 0, 0, 0, 0),
             (1, 0.8, 0.6, 0.4, 0.2, 0),
             (0, 0.2, 0.4, 0.6, 0.8, 1))
    np.testing.assert_array_equal(Mtest.shape, (n, D))
    np.testing.assert_almost_equal(Mtest, Mgood)
    for i in range(n):
        for j, u in enumerate(Ueval):
            assert N[i, 1](u) == pytest.approx(Mgood[i][j])

    # j = 2
    Mtest = N[:, 2](Ueval)
    Mgood = ((1, 0.64, 0.36, 0.16, 0.04, 0),
             (0, 0.32, 0.48, 0.48, 0.32, 0),
             (0, 0.04, 0.16, 0.36, 0.64, 1))
    np.testing.assert_array_equal(Mtest.shape, (n, D))
    np.testing.assert_almost_equal(Mtest, Mgood)
    for i in range(n):
        for j, u in enumerate(Ueval):
            assert N[i, 2](u) == pytest.approx(Mgood[i][j])


def test_BezierOrder3():
    p, n, D = 3, 4, 6
    U = SUF.UBezier(p=p)
    N = SBF.SplineBaseFunction(U)
    Ueval = np.linspace(0, 1, D)

    # j = 0
    Mtest = N[:, 0](Ueval)
    Mgood = ((0, 0, 0, 0, 0, 0),
             (0, 0, 0, 0, 0, 0),
             (0, 0, 0, 0, 0, 0),
             (1, 1, 1, 1, 1, 1))
    np.testing.assert_array_equal(Mtest.shape, (n, D))
    np.testing.assert_almost_equal(Mtest, Mgood)
    for i in range(n):
        for j, u in enumerate(Ueval):
            assert N[i, 0](u) == pytest.approx(Mgood[i][j])

    # j = 1
    Mtest = N[:, 1](Ueval)
    Mgood = ((0, 0, 0, 0, 0, 0),
             (0, 0, 0, 0, 0, 0),
             (1, 0.8, 0.6, 0.4, 0.2, 0),
             (0, 0.2, 0.4, 0.6, 0.8, 1))
    np.testing.assert_array_equal(Mtest.shape, (n, D))
    np.testing.assert_almost_equal(Mtest, Mgood)
    for i in range(n):
        for j, u in enumerate(Ueval):
            assert N[i, 1](u) == pytest.approx(Mgood[i][j])

    # j = 2
    Mtest = N[:, 2](Ueval)
    Mgood = ((0, 0, 0, 0, 0, 0),
             (1, 0.64, 0.36, 0.16, 0.04, 0),
             (0, 0.32, 0.48, 0.48, 0.32, 0),
             (0, 0.04, 0.16, 0.36, 0.64, 1))
    np.testing.assert_array_equal(Mtest.shape, (n, D))
    np.testing.assert_almost_equal(Mtest, Mgood)
    for i in range(n):
        for j, u in enumerate(Ueval):
            assert N[i, 2](u) == pytest.approx(Mgood[i][j])

    # j = 3
    Mtest = N[:, 3](Ueval)
    Mgood = ((1, 0.512, 0.216, 0.064, 0.008, 0),
             (0, 0.384, 0.432, 0.288, 0.096, 0),
             (0, 0.096, 0.288, 0.432, 0.384, 0),
             (0, 0.008, 0.064, 0.216, 0.512, 1))
    np.testing.assert_array_equal(Mtest.shape, (n, D))
    np.testing.assert_almost_equal(Mtest, Mgood)
    for i in range(n):
        for j, u in enumerate(Ueval):
            assert N[i, 3](u) == pytest.approx(Mgood[i][j])


def test_UniformP1N3():
    p, n, D = 1, 3, 11
    U = SUF.UUniform(p=p, n=n)
    N = SBF.SplineBaseFunction(U)
    Ueval = np.linspace(0, 1, D)

    # j = 0
    Mtest = N[:, 0](Ueval)
    Mgood = ((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
             (1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0),
             (0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1))
    np.testing.assert_array_equal(Mtest.shape, (n, D))
    np.testing.assert_almost_equal(Mtest, Mgood)
    for i in range(n):
        for j, u in enumerate(Ueval):
            assert N[i, 0](u) == pytest.approx(Mgood[i][j])

    # j = 1
    Mtest = N[:, 1](Ueval)
    Mgood = ((1, 0.8, 0.6, 0.4, 0.2, 0, 0, 0, 0, 0, 0),
             (0, 0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2, 0),
             (0, 0, 0, 0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1))
    np.testing.assert_array_equal(Mtest.shape, (n, D))
    np.testing.assert_almost_equal(Mtest, Mgood)
    for i in range(n):
        for j, u in enumerate(Ueval):
            assert N[i, 1](u) == pytest.approx(Mgood[i][j])


def test_UniformP1N4():
    p, n, D = 1, 4, 11
    U = SUF.UUniform(p=p, n=n)
    N = SBF.SplineBaseFunction(U)
    Ueval = np.linspace(0, 1, D)

    # j = 0
    Mtest = N[:, 0](Ueval)
    Mgood = ((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
             (1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0),
             (0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0),
             (0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1))
    np.testing.assert_array_equal(Mtest.shape, (n, D))
    np.testing.assert_almost_equal(Mtest, Mgood)
    for i in range(n):
        for j, u in enumerate(Ueval):
            assert N[i, 0](u) == pytest.approx(Mgood[i][j])

    # j = 1
    Mtest = N[:, 1](Ueval)
    Mgood = ((1, 0.7, 0.4, 0.1, 0, 0, 0, 0, 0, 0, 0),
             (0, 0.3, 0.6, 0.9, 0.8, 0.5, 0.2, 0, 0, 0, 0),
             (0, 0, 0, 0, 0.2, 0.5, 0.8, 0.9, 0.6, 0.3, 0),
             (0, 0, 0, 0, 0, 0, 0, 0.1, 0.4, 0.7, 1))
    np.testing.assert_array_equal(Mtest.shape, (n, D))
    np.testing.assert_almost_equal(Mtest, Mgood)
    for i in range(n):
        for j, u in enumerate(Ueval):
            assert N[i, 1](u) == pytest.approx(Mgood[i][j])


def test_UniformOrderP2N4():
    p, n, D = 2, 4, 11
    U = SUF.UUniform(p=p, n=n)
    N = SBF.SplineBaseFunction(U)
    Ueval = np.linspace(0, 1, D)

    # j = 0
    Mtest = N[:, 0](Ueval)
    Mgood = ((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
             (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
             (1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0),
             (0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1))
    np.testing.assert_array_equal(Mtest.shape, (n, D))
    np.testing.assert_almost_equal(Mtest, Mgood)
    for i in range(n):
        for j, u in enumerate(Ueval):
            assert N[i, 0](u) == pytest.approx(Mgood[i][j])

    # j = 1
    Mtest = N[:, 1](Ueval)
    Mgood = ((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
             (1, 0.8, 0.6, 0.4, 0.2, 0, 0, 0, 0, 0, 0),
             (0, 0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2, 0),
             (0, 0, 0, 0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1))
    np.testing.assert_array_equal(Mtest.shape, (n, D))
    np.testing.assert_almost_equal(Mtest, Mgood)
    for i in range(n):
        for j, u in enumerate(Ueval):
            assert N[i, 1](u) == pytest.approx(Mgood[i][j])

    # j = 2
    Mtest = N[:, 2](Ueval)
    Mgood = ((1, 0.64, 0.36, 0.16, 0.04, 0, 0, 0, 0, 0, 0),
             (0, 0.34, 0.56, 0.66, 0.64, 0.5, 0.32, 0.18, 0.08, 0.02, 0),
             (0, 0.02, 0.08, 0.18, 0.32, 0.5, 0.64, 0.66, 0.56, 0.34, 0),
             (0, 0, 0, 0, 0, 0, 0.04, 0.16, 0.36, 0.64, 1))
    np.testing.assert_array_equal(Mtest.shape, (n, D))
    np.testing.assert_almost_equal(Mtest, Mgood)
    for i in range(n):
        for j, u in enumerate(Ueval):
            assert N[i, 2](u) == pytest.approx(Mgood[i][j])


def test_UniformOrderP2N5():
    p, n, D = 2, 5, 11
    U = SUF.UUniform(p=p, n=n)
    N = SBF.SplineBaseFunction(U)
    Ueval = np.linspace(0, 1, D)

    # j = 0
    Mtest = N[:, 0](Ueval)
    Mgood = ((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
             (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
             (1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0),
             (0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0),
             (0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1))
    np.testing.assert_array_equal(Mtest.shape, (n, D))
    np.testing.assert_almost_equal(Mtest, Mgood)
    for i in range(n):
        for j, u in enumerate(Ueval):
            assert N[i, 0](u) == pytest.approx(Mgood[i][j])

    # j = 1
    Mtest = N[:, 1](Ueval)
    Mgood = ((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.),
             (1., 0.7, 0.4, 0.1, 0, 0, 0, 0, 0, 0, 0.),
             (0, 0.3, 0.6, 0.9, 0.8, 0.5, 0.2, 0, 0, 0, 0.),
             (0, 0, 0, 0, 0.2, 0.5, 0.8, 0.9, 0.6, 0.3, 0.),
             (0, 0, 0, 0, 0, 0, 0, 0.1, 0.4, 0.7, 1.))
    np.testing.assert_array_equal(Mtest.shape, (n, D))
    np.testing.assert_almost_equal(Mtest, Mgood)
    for i in range(n):
        for j, u in enumerate(Ueval):
            assert N[i, 1](u) == pytest.approx(Mgood[i][j])

    # j = 2
    Mtest = N[:, 2](Ueval)
    Mgood = ((1, 0.49, 0.16, 0.01, 0, 0, 0, 0, 0, 0, 0),
             (0, 0.465, 0.66, 0.585, 0.32, 0.125, 0.02, 0, 0, 0, 0),
             (0, 0.045, 0.18, 0.405, 0.66, 0.75, 0.66, 0.405, 0.18, 0.045, 0),
             (0, 0, 0, 0, 0.02, 0.125, 0.32, 0.585, 0.66, 0.465, 0),
             (0, 0, 0, 0, 0, 0, 0, 0.01, 0.16, 0.49,  1))
    np.testing.assert_array_equal(Mtest.shape, (n, D))
    np.testing.assert_almost_equal(Mtest, Mgood)
    for i in range(n):
        for j, u in enumerate(Ueval):
            assert N[i, 2](u) == pytest.approx(Mgood[i][j])


# def test_DerivativeVectors():
#     U = [0, 0, 1, 1]  # p = 1 and n = 2
#     N = SBF.SplineBaseFunction(U)
#     onediv = N.get_DerivativeVector(1)
#     np.testing.assert_almost_equal(onediv, np.array([1]))
#     with pytest.raises(ValueError):
#         N.get_DerivativeVector(0)
#     with pytest.raises(ValueError):
#         N.get_DerivativeVector(2)

#     U = [0, 0, 0, 1, 1, 1]  # p = 2 and n = 3
#     N = SBF.SplineBaseFunction(U)
#     onediv = N.get_DerivativeVector(2)
#     np.testing.assert_almost_equal(onediv, np.array([2, 2]))
#     onediv = N.get_DerivativeVector(1)
#     np.testing.assert_almost_equal(onediv, np.array([1]))
#     with pytest.raises(ValueError):
#         N.get_DerivativeVector(0)
#     with pytest.raises(ValueError):
#         N.get_DerivativeVector(3)

#     U = [0, 0, 0, 0, 1, 1, 1, 1]  # p = 3 and n = 4
#     N = SBF.SplineBaseFunction(U)
#     onediv = N.get_DerivativeVector(3)
#     np.testing.assert_almost_equal(onediv, np.array([3, 3, 3]))
#     onediv = N.get_DerivativeVector(2)
#     np.testing.assert_almost_equal(onediv, np.array([2, 2]))
#     onediv = N.get_DerivativeVector(1)
#     np.testing.assert_almost_equal(onediv, np.array([1]))
#     with pytest.raises(ValueError):
#         N.get_DerivativeVector(0)
#     with pytest.raises(ValueError):
#         N.get_DerivativeVector(4)

#     U = [0, 0, 0, 0, 1 / 2, 1, 1, 1, 1]  # p = 3 and n = 5
#     N = SBF.SplineBaseFunction(U)
#     onediv = N.get_DerivativeVector(3)

#     test = np.array([6, 3, 3, 6])
#     np.testing.assert_almost_equal(onediv, test)
#     onediv = N.get_DerivativeVector(2)
#     test = np.array([4, 2, 4])
#     np.testing.assert_almost_equal(onediv, test)
#     onediv = N.get_DerivativeVector(1)
#     test = np.array([2, 2])
#     np.testing.assert_almost_equal(onediv, test)
#     with pytest.raises(ValueError):
#         N.get_DerivativeVector(0)
#     with pytest.raises(ValueError):
#         N.get_DerivativeVector(4)

#     U = [0, 0, 0, 0, 1 / 3, 2 / 3, 1, 1, 1, 1]  # p = 3 and n = 6
#     N = SBF.SplineBaseFunction(U)
#     onediv = N.get_DerivativeVector(3)
#     test = np.array([9, 9 / 2, 3, 9 / 2, 9])
#     np.testing.assert_almost_equal(onediv, test)
#     onediv = N.get_DerivativeVector(2)
#     test = np.array([6, 3, 3, 6])
#     np.testing.assert_almost_equal(onediv, test)
#     onediv = N.get_DerivativeVector(1)
#     test = np.array([3, 3, 3])
#     np.testing.assert_almost_equal(onediv, test)
#     with pytest.raises(ValueError):
#         N.get_DerivativeVector(0)
#     with pytest.raises(ValueError):
#         N.get_DerivativeVector(4)

#     U = [0, 0, 0, 0, 1 / 5, 4 / 5, 1, 1, 1, 1]  # p = 3 and n = 6
#     N = SBF.SplineBaseFunction(U)
#     onediv = N.get_DerivativeVector(3)
#     test = np.array([15, 15 / 4, 3, 15 / 4, 15])
#     np.testing.assert_almost_equal(onediv, test)
#     onediv = N.get_DerivativeVector(2)
#     test = np.array([10, 5 / 2, 5 / 2, 10])
#     np.testing.assert_almost_equal(onediv, test)
#     onediv = N.get_DerivativeVector(1)
#     test = np.array([5, 5 / 3, 5])
#     np.testing.assert_almost_equal(onediv, test)
#     with pytest.raises(ValueError):
#         N.get_DerivativeVector(0)
#     with pytest.raises(ValueError):
#         N.get_DerivativeVector(4)

#     U = [0, 0, 0, 0, 1 / 5, 3 / 5, 1, 1, 1, 1]  # p = 3 and n = 6
#     N = SBF.SplineBaseFunction(U)
#     onediv = N.get_DerivativeVector(3)
#     test = np.array([15, 5, 3, 15 / 4, 15 / 2])
#     np.testing.assert_almost_equal(onediv, test)
#     onediv = N.get_DerivativeVector(2)
#     test = np.array([10, 10/ 3, 5 / 2, 5])
#     np.testing.assert_almost_equal(onediv, test)
#     onediv = N.get_DerivativeVector(1)
#     test = np.array([5, 5 / 2, 5 / 2])
#     np.testing.assert_almost_equal(onediv, test)
#     with pytest.raises(ValueError):
#         N.get_DerivativeVector(0)
#     with pytest.raises(ValueError):
#         N.get_DerivativeVector(4)
