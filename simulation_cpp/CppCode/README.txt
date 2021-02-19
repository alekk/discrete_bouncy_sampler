README for dbps.cpp
===================

This document assumes you are using linux.

You will need Armadillo installed (the code certainly works with version 8.4).

To compile use:
> ./comp

For help with the arguments, type:
> ./dbps

0. For the isotropic Gaussian (d=100) with delta=1.0 and kappa=0.2:
> ./dbps 0 100000 1.0 0.2 1000 1
This relates to the left plot of Figure 2.

1. For the an anisotropic Gaussian (d=50) with delta=2.0 and kappa=0.5;
> ./dbps 1 100000 2.0 0.5 1000 1
This relates to the right-hand plot of Figure 3.

3. To check out convergence from the tails in a light tailed target (d=50):
> ./dbps 2 1000 2 0.7 100
This relates to Section 4.3.
Gamma is set to 10; this relates to the right-hand plot of Figure 2.
Search the code for the parameter "gamma" to change it and relate to App C.1.


