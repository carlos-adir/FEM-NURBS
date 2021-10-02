import numpy as np


class SplineIntegralDouble:

    @staticmethod
    def validation_entry(j, k, sides):
        try:
            j = int(j)
        except Exception:
            raise TypeError("j must be an integer. Type = " + str(type(j)))

        if k is None:
            pass
        else:
            try:
                k = int(k)
            except Exception:
                raise TypeError("k must be an integer. Type = " + str(type(k)))
            if j < k:
                j, k = k, j

        if j < 2:
            if sides is None:
                pass
            else:
                pass
                # raise TypeError("You can't pass sides when j < 2")
        else:
            if isinstance(sides, np.ndarray):
                pass
            elif sides is None:
                raise ValueError("You need to pass the sides if j > 1")
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
            if j == 2 and sides.shape[1] != 1:
                errormsg = "The shape of sides must be (2, 1) when j = 2\n"
                errormsg += "Current = " + str(sides.shape)

                raise ValueError(errormsg)
            if j == 3 and sides.shape[1] != 2:
                errormsg = "The shape of sides must be (2, 2) when j = 3\n"
                errormsg += "Current = " + str(sides.shape)
                raise ValueError(errormsg)

    @staticmethod
    def getIntegralBase(j, k=None, sides=None):
        """
        We want to do the integral:

        [II_{i}] =  int_{u_{i}}^{u_{i+1}} [Nj]*[Nk]^T du

        As we know that N_{zp} = 0 forall u not in [u_{z}, u_{z+p+1})
        We need to compute only the base vector

        [Nj] = [N_{i-j, j}, N_{i-j+1, j}, ..., N_{i, j}]

        and the same for

        [Nk] = [N_{i-k, k}, N_{i-k+1, k}, ..., N_{i, k}]

        So, we will get a matrix of shape (j+1, k+1)
        And so, we know that there is a base matrix [M_{i}] 
        of same shape such that

        [II_{i}] = h_{i} * [M_{i}] = (u_{i+1}-u_{i}) * [M_{i}]

        that means

        [M_{i}] = (1/h_{i}) *  int_{u_{i}}^{u_{i+1}} [Nj]*[Nk]^T du

        This function returns [M_{i}]
        """

        SplineIntegralDouble.validation_entry(j, k, sides)
        if k is None:
            k = j
        sides = np.array(sides)
        if k > j:
            transpose = True
            j, k = k, j
        else:
            transpose = False

        if j == 0 and k == 0:
            M = SplineIntegralDouble.__computeM00()
        elif j == 1 and k == 0:
            M = SplineIntegralDouble.__computeM10()
        elif j == 1 and k == 1:
            M = SplineIntegralDouble.__computeM11()
        elif j == 2 and k == 0:
            M = SplineIntegralDouble.__computeM20(sides)
        elif j == 2 and k == 1:
            M = SplineIntegralDouble.__computeM21(sides)
        elif j == 2 and k == 2:
            M = SplineIntegralDouble.__computeM22(sides)
        elif j == 3 and k == 0:
            M = SplineIntegralDouble.__computeM30(sides)
        elif j == 3 and k == 1:
            M = SplineIntegralDouble.__computeM31(sides)
        elif j == 3 and k == 2:
            M = SplineIntegralDouble.__computeM32(sides)
        elif j == 3 and k == 3:
            M = SplineIntegralDouble.__computeM33(sides)
        else:
            errormsg = "We are able to get only j=0, 1, 2 and 3"
            raise NotImplementedError(errormsg)

        if transpose:
            return np.transpose(M)
        else:
            return M

    @staticmethod
    def __computeM00():
        M = np.ones((1, 1))
        return M

    @staticmethod
    def __computeM10():
        M = np.ones((2, 1)) / 2
        return M

    @staticmethod
    def __computeM11():
        M = (np.eye(2) + 1) / 6
        return M

    @staticmethod
    def __computeM20(sides):
        b = 1 + sides[0, 0]
        d = 1 + sides[1, 0]
        M = np.zeros((3, 1))
        M[1] += 1
        M[:2] += np.array([[1], [-1]]) / (3 * b)
        M[1:] += np.array([[-1], [1]]) / (3 * d)
        return M

    @staticmethod
    def __computeM21(sides):
        b = 1 + sides[0, 0]
        d = 1 + sides[1, 0]
        M = np.zeros((3, 2))
        M[1] += 1 / 2
        M[:2] += np.array([(3, 1),
                           (-3, -1)]) / (12 * b)
        M[1:] += np.array([(-1, -3),
                           (1, 3)]) / (12 * d)
        return M

    @staticmethod
    def __computeM22(sides):
        b = 1 + sides[0, 0]
        d = 1 + sides[1, 0]
        M = np.zeros((3, 3))
        II = 2 * np.eye(2) - 1
        M[1, 1] += 1
        M[:2, :2] += np.array(((0, 1),
                               (1, -2))) / (3 * b)
        M[1:, 1:] += np.array(((-2, 1),
                               (1, 0))) / (3 * d)
        M[:2, :2] += II / (5 * b**2)
        M[1:, 1:] += II / (5 * d**2)
        M[:2, 1:] -= II / (30 * b * d)
        M[1:, :2] -= II / (30 * b * d)
        return M

    @staticmethod
    def __computeM30(sides):
        M = np.zeros((4, 1))
        b = 1 + sides[0, 0]
        a = b + sides[0, 1]
        d = 1 + sides[1, 0]
        e = d + sides[1, 1]
        c = b + d - 1
        M[:2, 0] += np.array([1, -1]) / (4 * a * b)
        M[2:, 0] += np.array([-1, 1]) / (4 * d * e)
        factor = (d - b) * (2 * b * d - 1) / (4 * b * c * d)
        M[1:3, 0] += factor * np.array([1, -1]) + (1 / 2)
        return M

    @staticmethod
    def __computeM31(sides):
        M = np.zeros((4, 2))
        b = 1 + sides[0, 0]
        a = b + sides[0, 1]
        d = 1 + sides[1, 0]
        e = d + sides[1, 1]
        c = b + d - 1

        deno_center = 20 * b * c * d

        M[0] += np.array((4, 1)) / (20 * a * b)
        M[1, 0] += (10 * b * d**2 + b - 4 * d) / deno_center
        M[1, 0] += (- 4 * d * c) / (a * deno_center)
        M[1, 1] += (10 * b * d**2 - 10 * b * d + 4 * b - d) / deno_center
        M[1, 1] += (- d * c) / (a * deno_center)
        M[2, 0] += (10 * b**2 * d - 10 * b * d - b + 4 * d) / deno_center
        M[2, 0] += (- b * c) / (e * deno_center)
        M[2, 1] += (10 * b**2 * d - 4 * b + d) / deno_center
        M[2, 1] += (- 4 * b * c) / (e * deno_center)
        M[3] += np.array((1, 4)) / (20 * d * e)

        return M

    @staticmethod
    def __computeM32(sides):
        M = np.zeros((4, 3))
        b = 1 + sides[0, 0]
        a = b + sides[0, 1]
        d = 1 + sides[1, 0]
        e = d + sides[1, 1]
        c = b + d - 1

        deno_b2cd2 = 60 * b**2 * c * d**2
        deno_ab2d = 60 * a * b**2 * d
        deno_bd2e = 60 * b * d**2 * e

        M[0, 0] += 1 / (6 * a * b**2)
        M[0, 1] += (15 * b * d - b - 10 * d) / deno_ab2d
        M[0, 2] += 1 / (60 * a * b * d)

        M[1, 0] += (5 + 4 * c) / (12 * b * c)
        M[1, 0] += (b - 10 * d) / (60 * b**2 * c * d)
        M[1, 0] += (-1) / (6 * a * b**2)
        M[1, 1] += (10 * b**2 * d**2 * (6 * d - 5) + 40 * b**2 * d -
                    10 * b**2 - 20 * b * d**2 * (d + 1) + 10 * d**2) / deno_b2cd2
        M[1, 1] += (-15 * b**2 * d + b**2 - 15 * b * d**2 + 26 *
                    b * d - b + 10 * d**2 - 10 * d) / (60 * a * b**2 * c * d)
        M[1, 2] += -1 / (60 * a * b * d)
        M[1, 2] += (10 * b - 25 * b * d - d) / (60 * b * c * d**2)

        M[2, 0] += -1 / (60 * b * d * e)
        M[2, 0] += (10 * d - 25 * b * d - b) / (60 * b**2 * c * d)
        M[2, 1] += (10 * b**2 * d**2 * (6 * b - 5) - 20 * b**2 * d *
                    (b + 1) + 10 * b**2 + 40 * b * d**2 - 10 * d**2) / deno_b2cd2
        M[2, 1] += (-15 * b**2 * d + 10 * b**2 - 15 * b * d**2 +
                    26 * b * d - 10 * b + d**2 - d) / (60 * b * c * d**2 * e)
        M[2, 2] += -1 / (6 * d**2 * e)
        M[2, 2] += (5 + 4 * c) / (12 * c * d)
        M[2, 2] += (d - 10 * b) / (60 * b * c * d**2)

        M[3, 0] += 1 / (60 * b * d * e)
        M[3, 1] += (15 * b * d - 10 * b - d) / deno_bd2e
        M[3, 2] += 1 / (6 * d**2 * e)

        M[1, 0] += -1 / (3 * c)
        M[1, 2] += 1 / (3 * c)
        M[2, 0] += 1 / (3 * c)
        M[2, 2] += -1 / (3 * c)
        return M

    @staticmethod
    def __computeM33(sides):
        b = 1 + sides[0, 0]
        a = b + sides[0, 1]
        d = 1 + sides[1, 0]
        e = d + sides[1, 1]
        c = b + d - 1

        M = np.zeros((4, 4))
        II = 2 * np.eye(2) - 1
        VV = np.array(((0, 1, -1),
                       (1, -2, 1),
                       (-1, 1, 0)))

        consts = np.zeros(10)
        consts[0] = 1 / (7 * a**2 * b**2)
        consts[1] = 1 / (7 * d**2 * e**2)
        consts[2] = -1 / (140 * a * b * d * e)
        consts[3] = (5 * (d - b) + 9) / (40 * a * b * c)
        consts[3] += (b - 20 * d) / (140 * a * b**2 * c * d)
        consts[4] = (5 * (b - d) + 9) / (40 * c * d * e)
        consts[4] += (d - 20 * b) / (140 * b * c * d**2 * e)
        consts[5] = 1 / (8 * a * b)
        consts[6] = 1 / (8 * d * e)
        consts[7] = (b + d)**2 / (7 * b**2 * c**2 * d**2)
        consts[7] += (d**2 - d + 2) / (c**2)
        consts[7] -= 7 * (b + d) / (10 * b * c**2 * d)
        consts[7] -= 1 / (2 * b * c)
        consts[7] -= 3 / (10 * b * c**2 * d)
        consts[8] = (b + d)**2 / (7 * b**2 * c**2 * d**2)
        consts[8] += (b**2 - b + 2) / (c**2)
        consts[8] -= 7 * (b + d) / (10 * b * c**2 * d)
        consts[8] -= 1 / (2 * c * d)
        consts[8] -= 3 / (10 * b * c**2 * d)
        consts[9] = b**3 * d**3
        consts[9] -= b**3 * d**2 / 2
        consts[9] += b**3 * d / 4 - b**2 * d**3 / 2 - b**2 * d**2
        consts[9] += 9 * b**2 * d / 20 - b**2 / 7
        consts[9] += b * d**3 / 4 + 9 * b * d**2 / 20
        consts[9] += b * d / 70 - d**2 / 7
        consts[9] /= (b**2 * c**2 * d**2)

        M[:2, :2] += consts[0] * II
        M[2:, 2:] += consts[1] * II
        M[:2, 2:] += consts[2] * II
        M[2:, :2] += consts[2] * II
        M[:3, :3] += consts[3] * VV
        M[1:, 1:] += consts[4] * VV
        M[:3, :3] += consts[5] * np.array(((0, 1, 1),
                                           (1, -2, -1),
                                           (1, -1, 0)))
        M[1:, 1:] += consts[6] * np.array(((0, -1, 1),
                                           (-1, -2, 1),
                                           (1, 1, 0)))
        M[1, 1] += consts[7]
        M[2, 2] += consts[8]
        M[1, 2] += consts[9]
        M[2, 1] += consts[9]
        return M


def main():
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


if __name__ == "__main__":
    # main()
    pass
