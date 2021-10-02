import numpy as np


def transpose(M, diagonal=1):
    if not isinstance(diagonal, int):
        raise TypeError("Diagonal must be a number")
    if diagonal < 1 or 2 < diagonal:
        raise ValueError("Diagonal must be 1 or 2")
    try:
        M = np.array(M)
    except Exception:
        raise ValueError("M must be a matrix")
    if len(M.shape) == 1:
        return np.flip(M)
    elif len(M.shape) != 2:
        raise Exception("M must be 1D or 2D")

    if M.shape[0] == M.shape[1]:
        if diagonal == 1:
            newM = np.transpose(M)
        else:
            n = M.shape[0]
            newM = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    newM[i, j] = M[n - 1 - i, n - 1 - j]
    else:
        newM = np.zeros(M.shape)
        n, m = M.shape
        if diagonal == 1:
            errormsg = "Not possible tranpose M not-square when diagonal = 1"
            raise Exception(errormsg)
        elif diagonal == 2:
            for i in range(n):
                for j in range(m):
                    newM[i, j] = M[n - i - 1, m - j - 1]
        else:
            errormsg = "Not expected get here where diagonal = %s" % diagonal
            raise Exception(errormsg)

    return newM


def isSymetric(M, diagonal=1):
    tolerance = 1e-7
    if not isinstance(M, np.ndarray):
        raise TypeError("M in isSymetric must be a numpy array")
    if not isinstance(diagonal, int):
        raise TypeError("diagonal must be int")

    if len(M.shape) == 1:
        if diagonal != 1:
            raise ValueError("If M is 1D array, diagonal must be 1")
        return bool(np.all(np.abs(M - transpose(M)) < tolerance))
    elif len(M.shape) == 2:
        if diagonal != 1 and diagonal != 2:
            raise ValueError("If M is 2D array, diagonal must be 1 or 2")
        n, m = M.shape
        if diagonal == 1:
            return bool(np.all(np.abs(M - transpose(M, diagonal=1)) < tolerance))
        elif diagonal == 2:
            return bool(np.all(np.abs(M - transpose(M, diagonal=2)) < tolerance))
        else:
            raise Exception("Not possible get diagonal = " + str(diagonal))


def UBezier(p):
    return np.concatenate((np.zeros(p + 1), np.ones(p + 1)))


def UUniform(p, n):
    if n <= p:
        n = p + 1
    U1 = np.zeros(p)
    U2 = np.linspace(0, 1, n - p + 1)
    U3 = np.ones(p)
    return np.concatenate((U1, U2, U3))


def URandom(p, n):
    """
    We have n-p intervalos para determinar
    """
    if n <= p:
        n = p + 1
    intervals = np.random.rand(n - p)
    intervals /= np.sum(intervals)
    X = np.cumsum(intervals)
    X[-1] = 1
    return np.concatenate((np.zeros(p + 1), X, np.ones(p)))


def isValidU_withError(U):
    if isinstance(U, np.ndarray):
        pass
    elif isinstance(U, tuple):
        pass
    elif isinstance(U, list):
        pass
    else:
        errormsg = "U type must be a (tuple) or (list) or (numpy array):\n"
        errormsg += "    type(U) = " + str(type(U))
        raise TypeError(errormsg)

    try:
        U = np.array(U, dtype="float64")
    except Exception:
        errormsg = "Tried to convert U into numpy array. It was not possible."
        raise ValueError(errormsg)

    if len(U.shape) != 1:
        errormsg = "U must be a 1D array"
        raise ValueError(errormsg)

    if U.shape[0] < 4:
        errormsg = "The length of U must be at least 4"
        raise ValueError(errormsg)

    if np.any((U[1:] - U[:-1]) < 0):
        errormsg = "U must be ordened"
        raise ValueError(errormsg)

    if np.any(U < 0):
        errormsg = "All values inside U must be non-negative"
        raise ValueError(errormsg)

    if np.any(U > 1):
        errormsg = "All values inside U must be less than 1"
        raise ValueError(errormsg)

    if U[0] != 0:
        errormsg = "The frist value of the vector must be 0"
        raise ValueError(errormsg)
    if U[-1] != 1:
        errormsgr = "The last value of the vector must be 1"
        raise ValueError(errormsg)

    p = 0
    while U[p + 1] == 0 and U[-p - 2] == 1:
        p += 1

    if U[p + 1] == 0 or U[-p - 2] == 1:
        errormsg = "The quantity of 0 must be the same of 1"
        raise ValueError(errormsg)

    counter = {}
    for u in U:
        if not u in counter:
            counter[u] = 0
        counter[u] += 1
    for key in counter:
        if counter[key] > p + 1:
            errormsg = "Concentred middle nodes must be <= p+1: [%d/%d]" % (
                counter[key], p + 1)
            raise ValueError(errormsg)


def isValidU(U, throwError=False):
    try:
        isValidU_withError(U)
        return True
    except TypeError as e:
        if throwError:
            raise e
        return False
    except ValueError as e:
        if throwError:
            raise e
        return False


def getPfromU(U):
    p = 0
    while p < len(U):
        if U[p + 1] != 0:
            return p
        p += 1


def getNfromU(U):
    p = getPfromU(U)
    n = len(U) - p - 1
    return n


def transformUtoH(U, j=None):
    p = int(np.sum(U == 0) - 1)
    if j is None:
        j = p
    if not isinstance(j, int):
        errormsg = "j parameter must be an integer. Type(j) = " + str(type(j))
        raise TypeError(errormsg)
    elif j < 0:
        errormsg = "j must be non-negative"
        raise ValueError(errormsg)
    elif j > p:
        errormsg = "j must be <= p: (j, p) = " + str((j, p))
        raise ValueError(errormsg)
    n = len(U) - p - 1

    h = U[p + 1:n + 1] - U[p:n]
    if j > 1:
        z = np.zeros(j - 1)
        h = np.concatenate((z, h, z))
    return h


def cutHtoElementZ(H, z):
    j = int(np.sum(H == 0) // 2) + 1
    return np.copy(H[z:z + 2 * j - 1])


def transformHtoSides(hparted):
    if not len(hparted) % 2:
        raise ValueError("The quantity in hparted is not odd")
    p = len(hparted) // 2
    return np.array([np.flip(hparted[:p]), hparted[p + 1:]]) / hparted[p]


def isDiagonalDominant(M):
    if not isinstance(M, np.ndarray):
        raise TypeError("M is not a numpy array")
    if len(M.shape) != 2:
        raise ValueError("M must be a 2D array")
    if M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square matrix")
    n = M.shape[0]
    for i in range(n):
        soma = np.sum(np.abs(M[i])) - np.abs(M[i, i])
        if not (soma < np.abs(M[i, i])):
            return False
    return True


def getUplot(U, Ne=128):
    """
    Ne = Number of Elements

    """
    zero = 1e-11
    p = getPfromU(U)
    n = getNfromU(U)
    newU = np.linspace(U[p] + zero, U[p + 1] - zero, Ne + 1)
    for iiii in range(p + 1, n):
        a = U[iiii] + zero
        b = U[iiii + 1] - zero
        new_interval = np.linspace(a, b, Ne + 1)
        newU = np.concatenate((newU, new_interval))
    return newU
