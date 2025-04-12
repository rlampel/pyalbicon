import casadi as cs
import matplotlib.pyplot as plt
from . import create_bvp, initialization, lifting, plot_states


def newton(G, x_start, opts={}):
    """Newton's method.

    Keyword arguments:
        G       -- casadi function for which Newton's metho is performed
        x_start -- start point of Newton's method
        opts    -- dict that contains further option:
                    - TOL       -- final residual tolerance
                    - max_iter  -- maximum number of iterations
                    - verbose   -- print outputs or not
    """
    TOL = opts.get("TOL", 1.e-12)
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
        x = x + dx
        func_val = G(x)
        func_norm = cs.norm_2(func_val)
        func_arr += [func_norm]
        if (verbose):
            print("Iteration: ", counter, "\t", x)
            print("norm: ", func_norm)
    return x, func_arr


def auto_lifted_newton(problem, lift_init=None, s_init=None, opts={}):
    """Newton's method that adapt the lifting in every iteration.

    Keyword arguments:
        problem     -- instance of the problem class for a bvp
        lift_init   -- given initial lifting
        s_init      -- given initialization
        opts        -- dict that contains further option:
                    - TOL       -- final residual tolerance
                    - max_iter  -- maximum number of iterations
                    - verbose   -- print outputs or not
    """
    TOL = opts.get("TOL", 1.e-12)
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
        s_next = s_curr + ds
        counter += 1

        # determine best lifting
        graph_points = lifting.best_graph_lift(ode, R, time_points, s_next, time_points, s_dim)
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

    if (plot_iter):
        plt.pause(plot_delay)
        plt.close(fig)

    return func_arr

