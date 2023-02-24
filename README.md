# FEM-Nurbs

FEM-Nurbs uses [NURBs](nurbswiki) as base functions on [FEM](femwikipedia)

* ODE: 1D problems
* Poisson equation
* Static loaded structure


##  Motivation

FEM supposes that cutting (meshing) the domain in many parts (elements), then the assembly will behave near the same as the original object.
There are many elements possible, the most known are [polynomial elements][lagrangeelements] like: triangles, quadrilateral, tetrahedral, hexahedral and so on.

Although they are simple to apply, normally the founded field is only class C0 (continuous) which implies to derivated fields be discontinuous at the boundary between two elements.

## How to use this repository

It's not a library, and the application of NURBS depends on which problem it treats. 

Inside each folder there's an ```README``` to explain the content 

[nurbswiki]: https://en.wikipedia.org/wiki/Non-uniform_rational_B-spline
[femwikipedia]: https://en.wikipedia.org/wiki/Finite_element_method
[lagrangeelements]: https://defelement.com/elements/lagrange.html