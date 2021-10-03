import numpy as np
import femnurbs.SplineUsefulFunctions as SUF


class SplineDoubleIntegral:

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

        SplineDoubleIntegral.validation_entry(j, k, sides)
        if k is None:
            k = j
        sides = np.array(sides)
        if k > j:
            transpose = True
            j, k = k, j
        else:
            transpose = False

        if j == 0 and k == 0:
            M = SplineDoubleIntegral.__computeM00()
        elif j == 1 and k == 0:
            M = SplineDoubleIntegral.__computeM10()
        elif j == 1 and k == 1:
            M = SplineDoubleIntegral.__computeM11()
        elif j == 2 and k == 0:
            M = SplineDoubleIntegral.__computeM20(sides)
        elif j == 2 and k == 1:
            M = SplineDoubleIntegral.__computeM21(sides)
        elif j == 2 and k == 2:
            M = SplineDoubleIntegral.__computeM22(sides)
        elif j == 3 and k == 0:
            M = SplineDoubleIntegral.__computeM30(sides)
        elif j == 3 and k == 1:
            M = SplineDoubleIntegral.__computeM31(sides)
        elif j == 3 and k == 2:
            M = SplineDoubleIntegral.__computeM32(sides)
        elif j == 3 and k == 3:
            M = SplineDoubleIntegral.__computeM33(sides)
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

    @staticmethod
    def getIntegralAllDomain(U, j=None):
        SUF.isValidU(U)

        p = SUF.getPfromU(U)
        n = SUF.getNfromU(U)
        if j is None:
            j = p
        elif j < 0 or j > p:
            raise ValueError("You must pass 0 <= j <= p")
        h = SUF.transformUtoH(U, j=j)
        t = n - p
        M = np.zeros((n + j - p, n + j - p))
        for z in range(t):
            i = z + p
            hi = h[z + j - 1]
            if hi == 0:
                continue
            Hcut = SUF.cutHtoElementZ(h, z)
            Scut = SUF.transformHtoSides(Hcut)
            Mpp = SplineDoubleIntegral.getIntegralBase(j=j, k=j, sides=Scut)
            M[z:z + j + 1, z:z + j + 1] += hi * Mpp
        return M
