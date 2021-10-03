import numpy as np
import femnurbs.SplineUsefulFunctions as SUF


def binom(a, b):
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError("The binoms parameters are integers")
    if a < 0 or b < 0:
        raise ValueError("The values in binoms must be non-negatives")
    if a < b:
        raise ValueError("binom(a,b) must have 0 <= b <= a")
    if b == 0:
        return 1
    if a == 0 or a == 1:
        return 1
    result = int(1)
    for i in range(a, a - b, -1):
        result *= int(i)
    for i in range(2, b + 1):
        result //= int(i)
    return result


def getVr(p, w):
    Vr = np.zeros(w + 1)
    for i in range(w + 1):
        Vr[i] = binom(p + i, i)
    return Vr


def getTr(w):
    Tr = np.arange(w + 1)
    Tr = 2 * Tr - w
    return Tr


def getW(p, w, sides=None):
    if not isinstance(p, int):
        raise TypeError("The value of parameter 'p' must be integer")
    if not isinstance(w, int):
        raise TypeError("The value of parameter 'w' must be integer")
    if p < 1:
        raise ValueError("The value of parameter 'p' must be positive")
    if p > 3:
        raise NotImplementedError("Only implemented for 'p' = 1, 2 and 3")
    if w < 1:
        raise ValueError("The value of parameter 'w' must be positive")

    if p > 1:
        b = 1 + sides[0, 0]
        d = 1 + sides[1, 0]
    if p > 2:
        a = b + sides[0, 1]
        e = d + sides[1, 1]
        c = b + d - 1
    W = np.zeros((p + 1, w + 1))

    all_denominators = np.array([[6, 12, 20, 30, 42, 56],
                                 [12, 30, 60, 105, 168, 252],
                                 [20, 60, 140, 280, 504, 840]])
    den = all_denominators[p - 1, w - 1]
    Vr = getVr(p, w)
    Vl = np.flip(Vr)
    if p == 1:
        W[0] += Vl
        W[1] += Vr
    elif p == 2:
        W[1] += den / (w + 1)
        W[0] += Vl / b
        W[1] -= Vl / b
        W[1] -= Vr / d
        W[2] += Vr / d
    elif p == 3:
        Tr = getTr(w)
        Tl = np.flip(Tr)

        W[0] += Vl / (a * b)
        W[1] += Vr / (c * d)
        W[2] += Vl / (b * c)
        W[3] += Vr / (d * e)

        W[1] += binom(4 + w, 3) * (1 + (d - b) / c) / 2
        W[2] += binom(4 + w, 3) * (1 + (b - d) / c) / 2
        W[1] += binom(4 + w, 2) * Tl / (2 * c)
        W[2] += binom(4 + w, 2) * Tr / (2 * c)

        W[1] -= (a + c) * Vl / (a * b * c)
        W[2] -= (c + e) * Vr / (c * d * e)

    return W / den


def getWAllDomain(U, j, w):
    SUF.isValidU(U)

    p = SUF.getPfromU(U)
    n = SUF.getNfromU(U)

    if j < 0 or j > p:
        raise ValueError("You must pass 0 <= j <= p")
    h = SUF.transformUtoH(U, j=j)
    # print("h = ")
    # print(h)
    t = n - p
    W = np.zeros((n + j - p, t * w + 1))
    # print("M.shape = ", M.shape)
    for z in range(t):
        i = z + p
        hi = h[z + j - 1]
        # print("hi = ")
        # print(hi)
        if hi == 0:
            continue
        Hcut = SUF.cutHtoElementZ(h, z)
        # print("Hcut = ")
        # print(Hcut)
        Scut = SUF.transformHtoSides(Hcut)
        Wpp = getW(p=j, w=w, sides=Scut)
        W[z:z + j + 1, z * w: (z + 1) * w + 1] += hi * Wpp
    return W


def computeForceResultant(U, j, w, f):
    W = getWAllDomain(U, j, w)
    p = SUF.getPfromU(U)
    n = SUF.getNfromU(U)
    Ueval = SUF.rafineU(U, w)
    feval = np.zeros(Ueval.shape)
    for i, u in enumerate(Ueval):
        feval[i] = f(u)
    F = W @ feval
    return F


if __name__ == "__main__":
    print("binom(0, 0) = " + str(binom(0, 0)))
    print("binom(1, 0) = " + str(binom(1, 0)))
    print("binom(1, 0) = " + str(binom(1, 1)))
    print("binom(2, 0) = " + str(binom(2, 0)))
    print("binom(2, 1) = " + str(binom(2, 1)))
    print("binom(2, 2) = " + str(binom(2, 2)))
    print("binom(3, 0) = " + str(binom(3, 0)))
    print("binom(3, 1) = " + str(binom(3, 1)))
    print("binom(3, 2) = " + str(binom(3, 2)))
    print("binom(3, 3) = " + str(binom(3, 3)))
