import numpy as np
from numpy import linalg as la
import femnurbs.SplineUsefulFunctions as SUF
import femnurbs.SplineDoubleIntegral as SDI
import femnurbs.SplineBaseFunction as SBF
import femnurbs.SplineForceMatrix as SFM


def getMassMatrix(U):
    p = SUF.getPfromU(U)
    n = SUF.getNfromU(U)
    j = p - 1
    h = SUF.transformUtoH(U, j=j)
    M = SDI.SplineDoubleIntegral.getIntegralAllDomain(U, j=p - 1)

    ap = p / (U[p + 1:n + p] - U[1:n])
    Ap = np.zeros((n, n - 1))
    for i in range(n - 1):
        Ap[i, i] = -ap[i]
        Ap[i + 1, i] = ap[i]

    M = np.transpose(Ap @ M @ np.transpose(Ap))
    return M


def getBoundaryMatrix(U):
    p = SUF.getPfromU(U)
    n = SUF.getNfromU(U)
    h = SUF.transformUtoH(U)

    K = np.zeros((n, n))
    N = SBF.SplineBaseFunction(U)
    Nj0 = N[:, p - 1](0)[1:]
    Nj1 = N[:, p - 1](1)[1:]
    Np0 = N[:, p](0)
    Np1 = N[:, p](1)

    ap = p / (U[p + 1:n + p] - U[1:n])
    Ap = np.zeros((n, n - 1))
    for i in range(n - 1):
        Ap[i, i] = -ap[i]
        Ap[i + 1, i] = ap[i]

    J1 = np.transpose(Ap @ Nj1)
    J0 = np.transpose(Ap @ Nj0)
    T1 = np.tensordot(Np1, J1, axes=0)
    T0 = np.tensordot(Np0, J0, axes=0)
    F = T1 - T0
    return F


def getForceVector(U, f, w=6):
    p = SUF.getPfromU(U)
    n = SUF.getNfromU(U)
    F = SFM.computeForceResultant(U, p, w, f)
    return F


def solveFEM(M, F, dic):
    index_know = []
    values_know = []
    for key in dic:
        index_know.append(key)
    index_know = np.sort(np.array(index_know))
    for key in index_know:
        values_know.append(dic[key])

    index_unknown = []
    n = M.shape[0]
    for i in range(n):
        if i not in index_know:
            index_unknown.append(i)

    k, u = len(values_know), n - len(values_know)
    Muu = np.zeros((u, u))
    Mkk = np.zeros((k, k))
    Muk = np.zeros((u, k))
    Fu = np.zeros(u)
    Fk = np.zeros(k)
    Tk = np.copy(values_know)

    for i, x in enumerate(index_unknown):
        for j, y in enumerate(index_unknown):
            Muu[i, j] = M[x, y]
        for j, y in enumerate(index_know):
            Muk[i, j] = M[x, y]
        Fu[i] = F[x]
    for i, x in enumerate(index_know):
        for j, y in enumerate(index_know):
            Mkk[i, j] = M[x, y]
        Fk[i] = F[x]

    Fart = Fu - Muk @ Tk
    Tu = la.solve(Muu, Fart)
    T = np.zeros(n)
    for i, x in enumerate(index_know):
        T[x] = Tk[i]
    for i, x in enumerate(index_unknown):
        T[x] = Tu[i]
    return T


def findSolution(U, force, BC):
    Tl, Tr = BC
    n = SUF.getNfromU(U)
    know = {0: Tl, n - 1: Tr}

    M = getMassMatrix(U)
    K = getBoundaryMatrix(U)
    F = getForceVector(U, force)
    T = solveFEM(M - K, F, know)
    return T
