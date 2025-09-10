import casadi as cs


def residual_func(u, y, func_list, R):
    """
    Implementation of Algorithm 2 from Albersmeyer & Diehl to compute G(u,y)
    """
    num_lifts = len(func_list)
    u_dim = u.shape[0]
    G = cs.DM([])
    x_i = u
    for i in range(num_lifts):
        x_i = func_list[i](x_i)
        G_i = x_i - y[i * u_dim:(i + 1) * u_dim]
        G = cs.vertcat(G, G_i)
        x_i = y[i * u_dim:(i + 1) * u_dim]
    F = R(u, x_i)
    return G, F


def eval_lift(u, d, func_list, R):
    """
    Implementation of Algorithm 3 from Albersmeyer & Diehl to compute Z(u,d)
    """
    num_lifts = len(func_list)
    u_dim = u.shape[0]
    z = cs.DM([])
    x_i = u
    for i in range(num_lifts):
        x_i = func_list[i](x_i)
        z_i = x_i - d[i * u_dim:(i + 1) * u_dim]
        x_i = z_i
        z = cs.vertcat(z, z_i)
    return z


def condensed_newton(uk, xk, func_list, R, opts={}):
    """
    Implementation of Algorithm 4 from Albersmeyer & Diehl to compute Z(u,d)
    """
    TOL = opts.get("TOL", 1.e-8)
    max_iter = opts.get("max_iter", 50)
    verbose = opts.get("verbose", True)

    u_dim = uk.shape[0]
    counter = 0
    # compute residuals d and final evaluation F
    dk, Fk = residual_func(uk, xk, func_list, R)
    # print(uk, xk)

    # create casadi functions for Algorithms 2 and 3
    u = cs.MX.sym('u', u_dim)
    d = cs.MX.sym('u', xk.shape[0])

    z = eval_lift(u, d, func_list, R)
    Z = cs.Function('Z', [u, d], [z])
    # partial derivatives of Z
    dZdu = cs.Function('dZdu', [u, d], [cs.jacobian(Z(u, d), u)])
    dZdd = cs.Function('dZdd', [u, d], [cs.jacobian(Z(u, d), d)])
    # partial derivatives of final function
    dRdd = cs.Function('dRdd', [u, d], [cs.jacobian(R(u, Z(u, d)[-u_dim:]), d)])
    dRdu = cs.Function('dRdu', [u, d], [cs.jacobian(R(u, Z(u, d)[-u_dim:]), u)])

    res_norm = cs.norm_2(Fk) + cs.norm_2(dk)
    res_arr = [res_norm]
    if verbose:
        print(f"Iteration: {counter}\t{uk}")
        print("norm: ", res_norm)

    while (res_norm >= TOL and counter < max_iter):
        ak = -dZdd(uk, dk) @ dk
        Ak = dZdu(uk, dk)
        bk = Fk - dRdd(uk, dk) @ dk
        Bk = dRdu(uk, dk)

        # solve the condensed Newton system to obtain
        delta_uk = -cs.solve(Bk, bk)
        # perform the Newton step
        xk = xk + ak + Ak @ delta_uk
        uk = uk + delta_uk

        # call Algorithm 2 to obtain
        dk, Fk = residual_func(uk, xk, func_list, R)
        res_norm = cs.norm_2(Fk) + cs.norm_2(dk)
        res_arr += [res_norm]

        counter += 1
        if verbose:
            print(f"Iteration: {counter}\t{uk}")
            print("norm: ", res_norm)

    return cs.vertcat(uk, xk), res_arr  # uk, xk, Fk, dk


