import casadi as cs


def newton_path(G, x_start, lamb=1, opts={}):
    """Returns an array containing all Newton iterates for a given function
    with starting points and step size control.

    Keyword arguments:
        G   -- casadi function
        x_start -- start point of Newton's method
        lamb    -- step size control
    """
    TOL = opts.get("TOL", 1.e-8)
    max_iter = opts.get("max_iter", 500)

    x = x_start
    dim_x = x_start.shape[0]
    counter = 0
    y = cs.MX.sym('y', dim_x)

    func_norm = cs.norm_2(G(x))
    x_arr = [x[:2]]
    DG = cs.Function('J', [y], [cs.jacobian(G(y), y)], ['x_in'], ['J'])

    while (func_norm > TOL and counter < max_iter):
        dx = -cs.solve(DG(x), G(x))
        counter += 1
        x = x + dx * lamb
        func_norm = cs.norm_2(G(x))
        x_arr += [x[:2]]

    return x_arr

