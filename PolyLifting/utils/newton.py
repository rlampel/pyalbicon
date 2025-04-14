import casadi as cs


def newton(G, x_start, opts={}):
    """Newton's method.

    Keyword arguments:
        G       -- casadi function for which Newton's metho is performed
        x_start -- start point of Newton's method
        opts    -- dict that contains further option:
                    - TOL       -- final residual tolerance
                    - max_iter  -- maximum number of iterations
                    - verbose   -- print outputs or not
                    - log_type  -- "res" to return the list of residual norms
                                   else the iterates themselves are returned
    """
    TOL = opts.get("TOL", 1.e-12)
    max_iter = opts.get("max_iter", 100)
    verbose = opts.get("verbose", False)
    log_type = opts.get("log_type", "res")

    x = x_start
    dim_x = x_start.shape[0]
    counter = 0
    y = cs.MX.sym('y', dim_x)

    func_norm = cs.norm_2(G(x))
    func_arr = []
    if (log_type == "res"):
        func_arr += [func_norm]
    x_arr = [x]
    DG = cs.Function('J', [y], [cs.jacobian(G(y), y)], ['x_in'], ['J'])
    while (func_norm > TOL and counter < max_iter):
        dx = - cs.solve(DG(x), G(x))
        counter += 1
        x = x + dx
        x_arr += [x]
        func_norm = cs.norm_2(G(x))

        if (log_type == "res"):
            func_arr += [func_norm]
        else:
            func_arr += [dx]

        if (verbose):
            print("Iteration: ", counter, "\t", x)
    if (log_type == "res"):
        func_arr = [float(el) for el in func_arr]
    return x, func_arr

