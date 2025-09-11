import casadi as cs


def newton(G, x_start, opts={}):
    """Newton's method.

    Keyword arguments:
        G       -- casadi function for which Newton's method is performed
        x_start -- start point of Newton's method
        opts    -- dict that contains further option:
                    - TOL       -- final residual tolerance
                    - max_iter  -- maximum number of iterations
                    - verbose   -- print outputs or not
                    - log_type  -- "res" to return the list of residual norms
                                   else the iterates themselves are returned
    """
    TOL = opts.get("TOL", 1.e-8)
    max_iter = opts.get("max_iter", 100)
    verbose = opts.get("verbose", False)
    log_type = opts.get("log_type", "res")

    x = x_start
    counter = 0

    func_norm = cs.norm_2(G(x))
    func_arr = []
    if (log_type == "res"):
        func_arr += [func_norm]
    x_arr = [x]
    # newton_step = cs.rootfinder('newton_step', 'newton', G,
    #                             {'line_search': False, 'max_iter': 1, 'error_on_fail': False})
    newton_step = cs.rootfinder('newton_step', 'fast_newton', G,
                                {'max_iter': 1, 'error_on_fail': False})
    while (func_norm > TOL and counter < max_iter):
        x_new = newton_step(x)
        counter += 1
        x_arr += [x_new]
        func_norm = cs.norm_2(G(x_new))

        if (log_type == "res"):
            func_arr += [func_norm]
        else:
            dx = x_new - x
            func_arr += [dx]

        x = x_new
        if (verbose):
            print("Iteration: ", counter, "\t", x)
    if (log_type == "res"):
        func_arr = [float(el) for el in func_arr]
    return x, func_arr

