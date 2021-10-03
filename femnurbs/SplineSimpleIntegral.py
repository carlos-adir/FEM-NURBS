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
