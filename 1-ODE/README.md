# FEM-NURBS - ODE

   [ODE](odewikipedia) problems are given by

   $$\dfrac{dT}{dt} = f(t, T)$$

   At the interval $\left[a, \ b\right]$.
   We solve by FEM using BSpline as base functions.

   To compare and see how good the gotten solution is, we will use the analitical solution (when available) and the Runge-Kutta 4-th order.

----------------------------------------------------

## 1 - The basics

   The first kind of problem we treat is the linear first order ODE

   $$T' = f(t)$$

   With $f$ be any function of $t$ and initial condition $T(a) = T_a$. 

   To cover this kind of problem, see the notebook [1-basics.ipynb](linknotebookthebasics)

----------------------------------------------------

## 2 - First order linear

   This kind of problem we treat is the linear first order ODE

   $$T' = A \cdot T + B$$

   With $A$ and $B$ constants and initial condition $T(a) = T_a$. 

   To cover this kind of problem, see the notebook [2-firstorder.ipynb](linknotebookfirstorder)

----------------------------------------------------

## 3 - Second order linear

   This kind of problem is given by

   $$T'' + A\cdot T' + C \cdot T = D$$

   With $A$ and $B$ constants.

   The initial/boundary conditions may be diverse:

   * Dirichlet at begin interval: $T(a) = T_a$
   * Neumann at begin interval: $T'(a) = Q_a$
   * Robin at begin interval: $\alpha_a T(a) + \beta_a \cdot T'(a) = \gamma_a$
   * Dirichlet at end interval: $T(b) = T_b$
   * Neumann at end interval: $T'(b) = Q_b$
   * Robin at end interval: $\alpha_b T(b) + \beta_b \cdot T'(b) = \gamma_b$

   To cover this kind of problem, see the notebook [3-secondorder.ipynb](linknotebooksecondorder)

----------------------------------------------------

## 4 - System of $n$ ODEs:

   For this case we write a linear system like

   $$\mathbf{A} \cdot \mathbf{T}' + \mathbf{B} \cdot \mathbf{T} = \mathbf{C}$$

   For this case we need $n$ boundary conditions, which leads to all be Dirichlets'.


[odewikipedia]: https://en.wikipedia.org/wiki/Ordinary_differential_equation
[linknotebookthebasics]: https://github.com/carlos-adir/FEM-NURBS/blob/main/1-ODE/1-basics.ipynb
[linknotebookfirstorder]: https://github.com/carlos-adir/FEM-NURBS/blob/main/1-ODE/2-firstorder.ipynb
[linknotebooksecondorder]: https://github.com/carlos-adir/FEM-NURBS/blob/main/1-ODE/3-secondorder.ipynb
[linknotebooksystemnequations]: https://github.com/carlos-adir/FEM-NURBS/blob/main/1-ODE/4-systemequations.ipynb