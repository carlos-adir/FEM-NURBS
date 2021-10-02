import numpy as np

class SplineIntegralSimple:

    @staticmethod
    def validation_entry(p, sides):
        try:
            p = int(p)
        except Exception:
            raise TypeError("p must be an integer. Type = " + str(type(p)))

        if p < 2:
            if sides is None:
                pass
            else:
                pass
                # raise TypeError("You can't pass sides when p < 2")
        else:
            if isinstance(sides, np.ndarray):
                pass
            elif sides is None:
                raise ValueError("You need to pass the sides if p > 1")
            elif isinstance(sides, tuple) or isinstance(sides, list):
                try:
                    sides = np.array(sides)
                except Exception as e:
                    errormsg = "Tried to convert sides to numpy array. But it was not possible.\n"
                    errormsg += "Error: " + str(e)
                    raise ValueError(errormsg)
            else:
                errormsg = "The sides must be numpy array, or list or tuple\n"
                errormsg += "It's type is: " + str(type(sides))
                raise TypeError(errormsg)

            if len(sides.shape) != 2:
                raise ValueError("The side must be a matrix - Dimension 2")
            if sides.shape[0] != 2:
                raise ValueError("The side must be a matrix 2 x n")
            if np.any(sides < 0):
                errormsg = "All the values inside sides must be non-negatives"
                raise ValueError(errormsg)
            if p == 2 and sides.shape[1] != 1:
                raise ValueError(
                    "The shape of sides must be (2, 1) when p = 2")
            if p == 3 and sides.shape[1] != 2:
                raise ValueError(
                    "The shape of sides must be (2, 2) when p = 3")

    @staticmethod
    def getIntegralBase(p, sides=None):
        SplineIntegralSimple.validation_entry(p, sides)
        sides = np.array(sides)
        if p == 0:
            return SplineIntegralSimple.__computeV0()
        elif p == 1:
            return SplineIntegralSimple.__computeV1()
        elif p == 2:
            return SplineIntegralSimple.__computeV2(sides)
        elif p == 3:
            return SplineIntegralSimple.__computeV3(sides)
        else:
            raise NotImplementedError(
                "We are able to get only p=0, 1, 2 and 3")

    @staticmethod
    def __computeV0():
        return np.ones(1)

    @staticmethod
    def __computeV1():
        return np.ones(2) / 2

    @staticmethod
    def __computeV2(sides):
        b0 = sides[0, 0]
        d0 = sides[1, 0]
        b = b0 + 1
        d = 1 + d0

        V = np.array([1, -1, 0]) / (3 * b)
        V += np.array([0, 1, 0])
        V += np.array([0, -1, 1]) / (3 * d)
        return V

    @staticmethod
    def __computeV3(sides):
        a0 = sides[0, 1]
        b0 = sides[0, 0]
        d0 = sides[1, 0]
        e0 = sides[1, 1]
        a = a0 + b0 + 1
        b = b0 + 1
        c = b0 + 1 + d0
        d = 1 + d0
        e = 1 + d0 + e0

        V = np.array([1, -1, 0, 0]) / (4 * a * b)
        V += np.array([0, 1, 1, 0]) / 2
        V += np.array([0, 0, -1, 1]) / (4 * d * e)
        factor = (d - b) * (2 * b * d - 1) / (4 * b * c * d)
        V += factor * np.array([0, 1, -1, 0])
        return V

    @staticmethod
    def getVectorH(U):
        p = np.sum(U == 0) - 1
        n = len(U) - p - 1
        if p == 0:
            h = U[1:] - U[:-1]
            return h
        else:
            h = U[p + 1:n + 1] - U[p:n]
        if p > 1:
            z = np.zeros(p - 1)
            h = np.concatenate((z, h, z))
        return h

    @staticmethod
    def transformHintoSizes(hparted):
        if not len(hparted) % 2:
            raise ValueError("The quantity in hparted is not odd")
        p = len(hparted) // 2
        return np.array([np.flip(hparted[:p]), hparted[p + 1:]]) / hparted[p]

    @staticmethod
    def getIntegralAllDomain(U):
        """
        Sendo [Np] = [N_{0p}, N_{1p}, ..., N_{n-1, p}]
        Essa funcao retorna [V] tal que
        [V] = int_{0}^{1} [Np] du
        """

        p = np.sum(U == 0) - 1
        n = len(U) - p - 1
        h = SplineIntegralSimple.getVectorH(U)
        Vtot = np.zeros(n)

        for i in range(n - p):  # The quantity of intervals
            if p < 1:
                hi = h[i]
                hparted = [hi]
            else:
                hparted = h[i:i + 2 * p - 1]
                hi = hparted[p - 1]
            if hi == 0:
                continue

            sizes = SplineIntegralSimple.transformHintoSizes(hparted)
            Vp = SplineIntegralSimple.getIntegralBase(p, sizes)
            Vtot[i:i + p + 1] += hi * Vp
        return Vtot


if __name__ == "__main__":
    one = sp.Rational(1, 1)

    n = 30
    U = np.zeros(n + 1, dtype="object")
    for i in range(n + 1):
        U[i] = sp.symbols("u" + str(i))

    # U = np.linspace(0, 1, n+1)

    i = 8
    p = 3
    jmax = 3
    kmax = 0
    nivel = 4
    p = max([jmax, kmax])
    hi = sp.symbols("hi")

    a, b, c, d, e, f = sp.symbols("a b c d e f")
    # b = 0
    # d = 1

    # b = sp.Rational(1, 1) + 1
    # a = sp.Rational(0, 1) + b
    # d = sp.Rational(1, 1) + 1
    # e = sp.Rational(1, 1) + d

    ee = e
    # a = 1 + a0 + b0 + c0
    # b = 1 + b0 + c0
    # c = 1 + c0
    b0 = b - 1
    a0 = a - b
    # d = 1 + d0
    # e = 1 + d0 + e0
    # f = 1 + d0 + e0 + f0
    d0 = d - 1
    e0 = ee - d

    # b0 = 1
    # d0 = 1

    # a0 = sp.Rational(1, 1)
    # b0 = sp.Rational(1, 1)
    # d0 = sp.Rational(1, 1)
    # e0 = sp.Rational(1, 1)

    # U[i-2] = U[i-3] + a0*hi
    U[i - 1] = U[i - 2] + a0 * hi
    U[i] = U[i - 1] + b0 * hi
    U[i + 1] = U[i] + hi
    U[i + 2] = U[i + 1] + d0 * hi
    U[i + 3] = U[i + 2] + e0 * hi
    # U[i+4] = U[i+3] + f0*hi

    print(U)
    print("U[i] = ")
    print(U[i])

    u = sp.symbols("u")

    dU = np.zeros((n, p + 1), dtype="object")
    for bb in range(n - p - 1):
        for aa in range(1, p + 1):
            dU[bb, aa] = U[bb + aa] - U[bb]
            # dU[bb, aa] = sp.symbols("A_{%d%d}"%(bb, aa))

    def Nfunc(i, j, k, u):
        if j < 0:
            return 0
        if j == 0:
            if i == k:
                return 1
            else:
                return 0
        else:
            factor1 = (u - U[i]) / dU[i, j]
            factor2 = 1 + (U[i + 1] - u) / dU[i + 1, j]
            return factor1 * Nfunc(i, j - 1, k, u) + factor2 * Nfunc(i + 1, j - 1, k, u)

    def mdc(a, b):
        if a < b:
            a, b = b, a
        r = a % b
        while r != 0:
            a, b = b, r
            r = a % b
        return b

    def mmc(a, b):
        return a * b / mdc(a, b)

    h = sp.symbols("h")

    Nj = np.zeros(p + 1, dtype="object")
    # Nk = np.zeros(p+1, dtype="object")
    for z in range(p + 1):
        Nj[z] = Nfunc(i - p + z, jmax, i, u)
        Nj[z] = sp.sympify(Nj[z]).subs(u, h + U[i])
        # Nk[z] = Nfunc(i-p+z, kmax, i, u)
        # Nk[z] = sp.sympify(Nk[z]).subs(u, h + U[i])

    print("Nj = ")
    for z in range(p + 1):
        print("[%d/%d] = %s" % (z + 1, p + 1, str(Nj[z])))
    # print("Nk = ")
    # for z in range(p+1):
    #     print("[%d/%d] = %s" % (z+1, p+1, str(Nk[z])))

    A = np.zeros((p + 1, p + 1), dtype="object")
    for aa in range(p + 1):
        for bb in range(p + 1):
            A[aa, bb] = sp.symbols("A_{%d%d}" % (aa, bb))

    F = np.copy(Nj)
    # aa = 0
    # print("U = ", U)
    # print("U[i+aa] = ", U[i+aa])
    # print("U[i-p+1+aa] = ", U[i-p+aa+1])
    # for aa in range(p+1):
    #     F[aa] *= (U[i+aa]-U[i-p+1+aa])/A[p, aa]

    V = np.zeros((p + 1), dtype="object")
    for aa in range(p + 1):
        termo = 1
        integra = F[aa] * termo / hi
        V[aa] = sp.integrate(integra, (h, 0, hi))
        V[aa] = sp.simplify(V[aa])
        V[aa] /= termo

    if p == 3:
        V[1:3] *= (b + d - 1) / c
    for aa in range(len(V)):
        V[aa] = sp.simplify(V[aa])

    print("V = ")
    for aa, v in enumerate(V):
        print("V[%d] = %s" % (aa, str(v)))

    R = sp.Rational
    a0 = R(1, 5)
    b0 = R(1, 5)
    d0 = R(1, 5)
    e0 = R(1, 5)

    ava = a0 + b0 + 1
    bva = b0 + 1
    cva = b0 + 1 + d0
    dva = 1 + d0
    eva = 1 + d0 + e0

    fator = R(1, 1)
    for aa in range(len(V)):
        V[aa] = V[aa].subs(a, ava)
        V[aa] = V[aa].subs(b, bva)
        V[aa] = V[aa].subs(c, cva)
        V[aa] = V[aa].subs(d, dva)
        V[aa] = V[aa].subs(e, eva)
        num, den = sp.fraction(V[aa])
        fator = mmc(fator, den)

    V *= fator
    print("V = ", fator)
    for aa, v in enumerate(V):
        print("V[%d] = %s" % (aa, str(v)))
