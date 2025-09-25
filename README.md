# Python software for Automatic Lifting of BVP for Improved Convergence of Newton's method (PyAlbicon)
This software implements lifting algorithms for Newton's method applied to boundary value problems of the form
```math
    R(x, F(x)) = 0,
```
where $F$ consists of intermediate function evaluations.
Further details are described in [1]. It consists of three distinct parts:
- "PolyLifting" implements the greedy and enumerative lifting algorithms for polynomials as described in Section 4.1 of [1] as well as the Rosenbrock example from Section 4.2.
- "NNLifting" implements the adversarial attack on a neural network as described in Section 4.3 of [1].
- "ODELifting" implements the lifting via dynamic programming for ODE-based BVP as described in Section 4.4 and 4.5 of [1].

The main goal is to reduce the number of iterations of Newton's method required to solve these problems. Further information is given in the corresponding directories. The computations in [2] were performed using Python 3.13.5 on Ubuntu 24 running on an AMD Ryzen 7 4800h with 16GB of RAM. We noticed that using an Apple M2 CPU leads to slightly different results.

---

This software relies on CasADi [2], NumPy [3], and SciPy [4]. Additionally, PyTorch [5] is used to load the model weights of the neural network. A more detailed list can be found in the file 'requirements.txt'.

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

[5] Imambi, S., Prakash, K. B., & Kanagachidambaresan, G. R. (2021). PyTorch. In Programming with TensorFlow: solution for edge computing applications (pp. 87-104). Cham: Springer International Publishing.
