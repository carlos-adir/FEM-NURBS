# -*- coding: utf-8 -*-
"""
           @file: Exemple1-SolveHeat1D-Tknow.py
           @date: 03th October 2021
         @author: Carlos Adir (carlos.adir.leite@gmail.com)
          @title: Solving Heat Equation 1D With know temperature as BC
    @description: This code solves 1D problem of the heat equation given by

                    d^2 T
                    -----  + f(u) = 0
                     du^2

                  With the boundary condiditon that T(u=0) = Tl and T(u=1) = Tr
                  The given solution T is a Spline-Based Function, such that

                          n-1
                  T(u) =  sum  N_{ip}(u) * T_{i}
                         i = 0

                  As exemple, there's 
"""


import numpy as np
from numpy import linalg as la
import femnurbs.SolveHeatEquation1D as SHE
import femnurbs.SplineBaseFunction as SBF
import femnurbs.SplineUsefulFunctions as SUF
from matplotlib import pyplot as plt


def getAnaliticSolution(coefs, BC):
    """
    We have a function given by
        f(u) = a_{0} + a_{1} * u + ... + a_{m} * u^{m}
    And so, this functions receive the coeficients like:
        coefs = [a_{0}, a_{1}, ..., a_{m}]
    And returns a callable function ```Tana``` that solves the ODE
    So, we have that
        Tana(0) = Tl
        Tana(1) = Tr
    Where BC = (Tl, Tr)
    """
    Tl, Tr = BC
    m = len(coefs) - 1
    vector = np.zeros(3 + m)
    vector[0] = Tl
    vector[2:] = -np.array(coefs)
    for i, c in enumerate(coefs):
        vector[i + 2] /= (i + 1) * (i + 2)
    vector[1] = Tr - np.sum(vector)
    Tana = np.vectorize(lambda u: vector @ (u**np.arange(m + 3)))
    return Tana


def getForceFunctionFromCoefs(coefs):
    """
    We have a ```force function``` called f(u) in the EDO:
        f(u) = a_{0} + a_{1} * u + ... + a_{m} * u^{m}
    And so, this functions receive the coeficients like:
        coefs = [a_{0}, a_{1}, ..., a_{m}]
    And returns the callable function ```f(u)```
    """
    vector = np.array(coefs)
    return np.vectorize(lambda u: vector @ (u**np.arange(len(coefs))))


def main():
    # Boundary temperatures
    Tl = 1
    Tr = 0.95

    # Mesh
    p = 2
    n = 7
    U = SUF.UUniform(p=p, n=n)
    print("With (p, n) = " + str((p, n)))
    print("Mesh Vector")
    print("U = " + str(U))

    ##################################
    #       Calculating Matrix       #
    ##################################
    M = SHE.getMassMatrix(U)
    print("Mass Matrix: M.shape = " + str(M.shape))

    K = SHE.getBoundaryMatrix(U)
    print("Boundary Matrix: K.shape = " + str(K.shape))

    ##################################
    #         Force function         #
    ##################################
    coefs_polynomial = [-1.2, 6]  # f(u) = -1.2 + 6u
    coefs_polynomial = [-1.2, 6, -12]  # f(u) = -1.2 + 6u - 12u^2
    coefs_polynomial = [-1.2, 6, -12, 8]  # f(u) = -1.2 + 6u - 12u^2 + 8u^3
    force = getForceFunctionFromCoefs(coefs_polynomial)
    Tana = getAnaliticSolution(coefs_polynomial, (Tl, Tr))

    F = SHE.getForceVector(U, force)
    print("Force Vector: F.shape = " + str(F.shape))

    ##################################
    #         Solving system         #
    ##################################
    print("............................")
    print(".      SOLVING SYSTEM      .")
    print("............................")
    know = {0: Tl, n - 1: Tr}
    Tnodes = SHE.solveFEM(M - K, F, know)
    print("Vector of nodes of T =", Tnodes.shape)
    print(Tnodes)

    ##################################
    #        Plotting results        #
    ##################################
    Uplot = SUF.getUplot(U)
    N = SBF.SplineBaseFunction(U)
    Nvals = N(Uplot)

    Tanaplot = Tana(Uplot)
    Tfemplot = Tnodes @ Nvals

    Unodes = np.zeros(n)
    for i in range(n):
        maximal = np.max(Nvals[i, :])
        index = np.where(Nvals[i, :] == maximal)
        Unodes[i] = Uplot[index]

    error = la.norm(Tfemplot - Tanaplot)
    print("Error of solution = " + str(error))

    plt.figure()
    plt.plot(Uplot, Tfemplot, label="FEM")
    plt.plot(Uplot, Tanaplot, label="analitic")
    plt.plot(Unodes, Tnodes, color="k",
             label="T nodes", ls="dotted", marker="o")
    plt.xlabel(r"$u$")
    plt.ylabel(r"$T$")
    plt.title(r"Solution of equation $\nabla^2 T + f = 0$")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
