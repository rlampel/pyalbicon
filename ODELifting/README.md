# Lifted BVP Solver



## Getting started

To solve the benchmark problems T19-T33 from [1] in a row, simply run 
```
python benchmark.py
```
The benchmark compares the following lifting approaches from [2]:
- <b>no lifting</b>
- <b>graph lifting</b> (determines the lifting with the best residual contraction once at the start using Algorithm 4 from [2])
- <b>automatic lifting</b> (determines the lifting with the best residual contraction in every iteration as described in Section 3.4.3 of [2])
- <b>heuristic + auto</b> (combines the initial values determined via the heuristic with the automatic lifting)

By default only the convergence with respect to the residuals are plotted. Further options such as plotting the heatmaps shown in Figure 12 of [2] can be adjusted at the top of the file.

---

To plot the Newton path for the Lotka-Volterra BVP from Example 2 in [2], run
```
python lotka_path.py
```
The damping parameter $\lambda$ and the custom initialization can be changed at the top of the file.

---

[1]: Mazzia, F., Cash, J.R.: A Fortran Test Set for Boundary Value Problem Solvers, vol. 1648 (2015)

[2]: Lampel, R., Sager, S.: On liftings that improve convergence properties of Newton’s Method for Boundary Value Optimization Problems