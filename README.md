# BVP Lifting Python
This software implements the lifting algorithms for Newton's method described in [1]. It consists of three distinct parts:
- "NNLifting" implements the adversarial attack on a neural network as described in Section 4.4 of [1].
- "ODELifting" implements the lifting via dynamic programming for ODE-based BVP as described in Section 4.5 of [1].
- "PolyLifting" implements the greedy and enumerative lifting algorithms for polynomials as described in Section 4.2 of [1] as well as the Rosenbrock example from section 4.3.

Further information is given in the corresponding directories.

---
[1]: [Lampel, R., Sager, S.: "On liftings that improve convergence properties of Newton’s Method for Boundary Value Optimization Problems"](https://optimization-online.org/?p=29392) 