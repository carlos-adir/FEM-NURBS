# FEM-NURBS - Poisson

   [Poisson's equation](poissonequation) is given by

   $$\nabla^2 T = f(\mathbf{x})$$

   At the domain $\Omega$ and boundary conditions.

   $$\begin{align}\partial \Omega & = \Gamma_{d} \cup \Gamma_{n} \cup \Gamma_{r} \\ = & \Gamma_{d} \cap \Gamma_{n} = \Gamma_{d} \cap \Gamma_{r} = \Gamma_{n} \cap \Gamma_{r} \end{align}$$

   We solve by FEM using BSpline as base functions.

   To compare and see how good the gotten solution is, we will use the analitical solution (when available).

----------------------------------------------------

## 1 - The Laplace Equation

   The first kind of problem we treat is the Laplace Equation:

   $$\nabla^2 T = 0$$

   To cover this kind of problem, see the notebook [1-laplace-basic.ipynb](linknotebooklaplacebasic)

----------------------------------------------------

## 2 - First order linear

   This kind of problem we treat is the linear first order ODE

   $$T' = A \cdot T + B$$

   With $A$ and $B$ constants and initial condition $T(a) = T_a$. 

   To cover this kind of problem, see the notebook [2-firstorder.ipynb](linknotebookfirstorder)
ets'.


[poissonequation]: https://en.wikipedia.org/wiki/Poisson%27s_equation
[linknotebookthebasics]: https://github.com/carlos-adir/FEM-NURBS/blob/main/1-ODE/1-basics.ipynb
[linknotebookfirstorder]: https://github.com/carlos-adir/FEM-NURBS/blob/main/1-ODE/2-firstorder.ipynb
[linknotebooksecondorder]: https://github.com/carlos-adir/FEM-NURBS/blob/main/1-ODE/3-secondorder.ipynb
[linknotebooksystemnequations]: https://github.com/carlos-adir/FEM-NURBS/blob/main/1-ODE/4-systemequations.ipynb