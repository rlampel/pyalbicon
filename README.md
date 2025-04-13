# Python software for Automatic Lifting of BVP for Improved Convergence of Newton's method (PyAlbicon)
This software implements lifting algorithms for Newton's method applied to boundary value problems of the form
$$
    R(x, F(x)) = 0,
$$
where $F$ consists of intermediate function evaluations.
Further details are described in [1]. It consists of three distinct parts:
- "NNLifting" implements the adversarial attack on a neural network as described in Section 4.4 of [1].
- "ODELifting" implements the lifting via dynamic programming for ODE-based BVP as described in Section 4.5 of [1].
- "PolyLifting" implements the greedy and enumerative lifting algorithms for polynomials as described in Section 4.2 of [1] as well as the Rosenbrock example from Section 4.3.

Further information is given in the corresponding directories.

---

This software relies on CasADi [2], NumPy [3], and SciPy [4].

---
[1]: [Lampel, R., Sager, S.: "On liftings that improve convergence properties of Newton’s Method for Boundary Value Optimization Problems"](https://optimization-online.org/?p=29392) 

[2]: Andersson, J.A.E., Gillis, J., Horn, G., Rawlings, J.B., Diehl, M.: CasADi – A
software framework for nonlinear optimization and optimal control. Mathematical
Programming Computation 11(1), 1–36 (2019)

[3] Harris, C.R., Millman, K.J., Walt, S.J., Gommers, R., Virtanen, P., Cournapeau,
D., Wieser, E., Taylor, J., Berg, S., Smith, N.J., Kern, R., Picus, M., Hoyer,
S., Kerkwijk, M.H., Brett, M., Haldane, A., Rı́o, J.F., Wiebe, M., Peterson,
P., Gérard-Marchant, P., Sheppard, K., Reddy, T., Weckesser, W., Abbasi, H.,
Gohlke, C., Oliphant, T.E.: Array programming with NumPy. Nature 585(7825),
357–362 (2020)

[4] Gommers, R., Virtanen, P., Haberland, M., Burovski, E., Reddy, T., Weckesser,
W., Oliphant, T.E., Cournapeau, D., Nelson, A., alexbrc, Roy, P., Peterson, P.,
Polat, I., Wilson, J., endolith, Mayorov, N., Walt, S., Colley, L., Brett, M., Lax-
alde, D., Larson, E., Sakai, A., Millman, J., Bowhay, J., Lars, peterbell10, Carey,
C., Mulbregt, P., eric-jones, Striega, K.: Scipy/scipy: SciPy 1.15.0