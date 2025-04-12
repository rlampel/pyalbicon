import numpy as np
import casadi as cs
from . import initialize, create_poly, newton


def greedy_start(coeffs, x_in, lift_degree=2, lift_type=""):
    """Create a lifted polynomial out of given coefficients in a greedy manner.

    Keyword arguments:
        coeffs  -- coefficients of the polynomial
        x_in    -- start point
        lift_degree -- degree of component functions
        lift_type   -- type of lifting (multilinear of rational exponents)
    """
    # lift at all possible points
    poly_dim = len(coeffs)
    max_degree = poly_dim - 1
    G_all = create_poly.create_lifted_poly(coeffs, lift_degree, lift_type=lift_type)
    lift_start = initialize.initialize_auto(x_in, poly_dim, lift_degree)
    opts = {"max_iter": 1}
    lift_step, _ = newton.newton(G_all, lift_start, opts)

    num_lifts = int(np.round(np.log(max_degree) / np.log(lift_degree), 2)) + 1
    lift_in = [0] * num_lifts
    g_lift = greedy_lift(G_all, lift_step, lift_in, poly_dim, lift_degree)

    G = create_poly.create_lifted_poly(coeffs, lift_degree, g_lift, lift_type=lift_type)
    s_out = lift_step[:2]
    for i in range(1, num_lifts):
        if (g_lift[i]):
            s_out = cs.vertcat(s_out, lift_step[2 * i:2 * i + 2])
    return G, s_out


def enumerate_start(coeffs, x_in, lift_degree=2, lift_type=""):
    """Create a lifted polynomial out of given coefficients by enumerating all possibilities.

    Keyword arguments:
        coeffs  -- coefficients of the polynomial
        x_in    -- start point
        lift_degree -- degree of component functions
        lift_type   -- type of lifting (multilinear of rational exponents)
    """
    # lift at all possible points
    poly_dim = len(coeffs)
    max_degree = poly_dim - 1
    G_all = create_poly.create_lifted_poly(coeffs, lift_degree, lift_type=lift_type)
    lift_start = initialize.initialize_auto(x_in, poly_dim, lift_degree)

    # compute states after one Newton step
    opts = {"max_iter": 1}
    lift_step, _ = newton.newton(G_all, lift_start, opts)
    print(" lift_step: ", lift_step)

    # compute best lifting by enumerating all choices
    num_lifts = int(np.round(np.log(max_degree) / np.log(lift_degree), 2)) + 1
    e_lift = enumerate_lift(G_all, lift_step, num_lifts, poly_dim, lift_degree)

    G = create_poly.create_lifted_poly(coeffs, lift_degree, e_lift, lift_type=lift_type)
    s_out = lift_step[:2]
    for i in range(1, num_lifts):
        if (e_lift[i]):
            s_out = cs.vertcat(s_out, lift_step[2 * i:2 * i + 2])

    return G, s_out


def num_to_binary(num, output_length):
    """Convert a number into a list of binary values.

    Keyword arguments:
        num --  number to convert
        output_length   --  length of the output array
    """
    # get binary representation
    res = [int(x) for x in bin(num)[2:]] + [0]
    res.reverse()
    curr_len = len(res)
    if (curr_len > output_length):
        raise ValueError("output length is too short")
    else:
        res += [0] * (output_length - curr_len)
    return res


def enumerate_lift(poly, x_in, num_lifts, num_coeffs, lift_degree=2):
    """Determine the best lifting by enumerating all possibilities.

    Keyword arguments:
        poly    -- polynomial lifted at all possible points
        x_in    -- initial values of intermediate variables
        num_lifts --  number of lifting points
        output_length   --  length of the output array
    """
    lift_in = [0] * num_lifts
    x_lift_in = initialize.initialize_auto(x_in, num_coeffs, lift_degree, lift_in)
    best_norm = cs.norm_2(poly(x_lift_in))
    best_lift = lift_in.copy()

    for i in range(2**(num_lifts - 1)):
        curr_lift = num_to_binary(i, num_lifts)
        x_temp = initialize.initialize_auto(x_in, num_coeffs, lift_degree, curr_lift)
        curr_norm = cs.norm_2(poly(x_temp))
        if (curr_norm <= best_norm):
            best_lift = curr_lift.copy()
            best_norm = curr_norm

    return best_lift


def greedy_lift(poly, x_in, lift_in, num_coeffs, lift_degree=2):
    """Determine a greedy lifting.

    Keyword arguments:
        poly    -- polynomial lifted at all possible points
        x_in    -- initial values of intermediate variables
        num_lifts --  number of lifting points
        output_length   --  length of the output array
    """
    greedy_lift = lift_in.copy()

    x_lift_in = initialize.initialize_auto(x_in, num_coeffs, lift_degree, lift_in)
    best_norm = cs.norm_2(poly(x_lift_in))

    for i in range(1, len(lift_in)):
        lift_temp = greedy_lift.copy()
        # change lifting at current point
        if greedy_lift[i]:
            lift_temp[i] = 0
        else:
            lift_temp[i] = 1

        x_temp = initialize.initialize_auto(x_in, num_coeffs, lift_degree, lift_temp)
        curr_norm = cs.norm_2(poly(x_temp))

        if (curr_norm < best_norm):
            # change lifting if current is better
            greedy_lift[i] = lift_temp[i]

    return greedy_lift

