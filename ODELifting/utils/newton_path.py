import casadi as cs


def newton_path(G, x_start, lamb=1, opts={}):
    """Returns an array containing all Newton iterates for a given function
    with step size control.

    Keyword arguments:
        G   -- casadi function
        x_start -- start point of Newton's method
        lamb    -- step size control
    """
    TOL = opts.get("TOL", 1.e-8)
    max_iter = opts.get("max_iter", 500)

    x = x_start
    counter = 0

    func_norm = cs.norm_2(G(x))
    x_arr = [x[:2]]
    newton_step = cs.rootfinder('one_step', 'newton', G,
                                {'line_search': False, 'max_iter': 1, 'error_on_fail': False})

    while (func_norm > TOL and counter < max_iter):
        x_new = newton_step(x)
        dx = x_new - x
        counter += 1
        x = x + dx * lamb
        func_norm = cs.norm_2(G(x))
        x_arr += [x[:2]]

    return x_arr

