import casadi as cs
import numpy as np
import matplotlib.pyplot as plt
from . import create_bvp, initialization, lifting, plot_states


def trust_region(G, x, dx, start_mu=1, TOL=1.e-6, verbose=False):
    """Deuflhard's residual-based trust region algorithm.

    Keyword arguments:
        G   -- casadi function
        x   -- starting point
        dx  -- current step
        start_mu   -- initial estimate for residual contraction parameter
        TOL -- final tolerance
        verbose -- print additional information
    """
    lam_min = 1.e-11
    mu = start_mu
    if (np.isnan(mu)):
        mu = np.inf
        lam = 1
    else:
        lam = np.min([1, mu])
    new_lam = 0
    flag = False

    while (True):
        if (verbose):
            print("trust region with lam = ", lam)
        if (np.abs(lam) < lam_min):
            print("Step size too small!")
            # flag = True
            break
        x_next = x + lam * dx
        # compute monitoring quantities
        theta = cs.norm_2(G(x_next)) / cs.norm_2(G(x))
        if (np.isnan(theta)):
            theta = np.inf

        mu = 0.5 * lam**2 * cs.norm_2(G(x)) / cs.norm_2(G(x_next) - (1 - lam) * G(x))
        mu = float(mu)
        if (np.isnan(mu)):
            mu = np.inf

        if (theta >= 1 - lam / 4):
            new_lam = np.min(np.array([mu, 0.5 * lam]).flatten())
            lam = new_lam
        else:
            break

    return lam, mu, flag


def newton(G, x_start, opts={}):
    """More accurate version of Newton's method that computes and inverts the Jacobian.

    Keyword arguments:
        G       -- casadi function for which Newton's method is performed
        x_start -- start point of Newton's method
        opts    -- dict that contains further option:
                    - TOL       -- final residual tolerance
                    - max_iter  -- maximum number of iterations
                    - verbose   -- print outputs or not
    """
    TOL = opts.get("TOL", 1.e-8)
    max_iter = opts.get("max_iter", 500)
    verbose = opts.get("verbose", False)

    x = x_start
    dim_x = x_start.shape[0]
    counter = 0
    y = cs.MX.sym('y', dim_x)

    func_val = G(x)
    func_norm = cs.norm_2(func_val)
    func_arr = [func_norm]
    DG = cs.Function('J', [y], [cs.jacobian(G(y), y)], ['x_in'], ['J'])

    while (func_norm > TOL and counter < max_iter):
        dx = -cs.solve(DG(x), func_val)
        counter += 1
        # lam, _, _ = trust_region(G, x, dx)
        lam = 1
        x = x + lam * dx
        func_val = G(x)
        func_norm = cs.norm_2(func_val)
        func_arr += [func_norm]
        if (verbose):
            print("Iteration: ", counter, "\t", x)
            print("norm: ", func_norm)
    return x, func_arr


def auto_lifted_newton(problem, lift_init=None, s_init=None, opts={}):
    """Newton's method that adapts the lifting in every iteration.

    Keyword arguments:
        problem     -- instance of the problem class for a bvp
        lift_init   -- given initial lifting
        s_init      -- given initialization
        opts        -- dict that contains further option:
                    - TOL       -- final residual tolerance
                    - max_iter  -- maximum number of iterations
                    - verbose   -- print outputs or not
                    - plot_iter -- plot the states in every iteration
                    - plot_delay -- how long to show the plot in every iteration
    """
    TOL = opts.get("TOL", 1.e-8)
    max_iter = opts.get("max_iter", 500)
    verbose = opts.get("verbose", False)
    plot_iter = opts.get("plot", True)
    plot_delay = opts.get("plot_delay", 1)

    if (plot_iter):
        fig = plt.figure(figsize=(10, 7))

    # get ode and boundary function
    ode = problem.get_ode()
    R = problem.get_boundary_fct()

    # get initial values and dimension
    init = problem.get_init()
    s_dim = init["s_dim"]

    # get time grid
    min_t, max_t, num_lifts = problem.get_grid_details()
    grid = {}
    time_points = [min_t + (max_t - min_t) * i / num_lifts for i in range(num_lifts + 1)]
    grid["time"] = time_points

    # compute all intermediate values
    if (lift_init is None):
        grid["lift"] = [0 for i in range(len(time_points))]
    else:
        grid["lift"] = lift_init

    if (s_init is None):
        init["sol"] = initialization.initialize(init, grid, ode)
    else:
        init["sol"] = s_init
        init["sol"] = initialization.initialize(init, grid, ode)

    # lift function at all possible points
    grid["lift"] = [1 for i in range(len(time_points))]
    B_lift_all = create_bvp.create_bvp(ode, R, grid, s_dim)

    S = cs.MX.sym("S", s_dim * len(time_points))
    DB_lift_all = cs.Function('DB', [S], [cs.jacobian(B_lift_all(S), S)], ["S"], ["grad"])

    func_val = B_lift_all(init["sol"])
    func_norm = cs.norm_2(func_val)
    func_arr = [func_norm]
    s_curr = init["sol"]
    counter = 0

    while (func_norm > TOL and counter < max_iter):
        # perform Newton step with all lifting points
        ds = - cs.solve(DB_lift_all(s_curr), func_val)
        # lam, _, _ = trust_region(B_lift_all, s_curr, ds)
        lam = 1
        s_next = s_curr + ds * lam
        counter += 1

        # determine best lifting
        graph_points = lifting.best_graph_lift(ode, R, time_points, s_next, time_points, s_dim,
                                               parallel=False)
        graph_lift = initialization.convert_lifting(graph_points, time_points)
        s_next = initialization.select_states(s_next, s_dim, graph_lift)

        # initialize other points automatically
        grid["lift"] = graph_lift
        init["sol"] = s_next
        s_curr = initialization.initialize(init, grid, ode)

        func_val = B_lift_all(s_curr)
        func_norm = cs.norm_2(func_val)
        func_arr += [func_norm]
        if (plot_iter):
            plt.clf()
            plot_states.plot_segmented(fig, init, grid, ode)

            plt.title("Iteration " + str(counter), fontsize=22)
            plt.pause(plot_delay)

        if (verbose):
            print("Iteration: ", counter, "\t current best lift: ", graph_points)
            print("current values: ", s_curr)
            print(" current norm: ", func_norm)

    if (plot_iter):
        plt.pause(plot_delay)
        plt.close(fig)

    return func_arr

