import numpy as np
import casadi as cs
from . import complex_utils


def initialize_auto(x0, num_coeffs, lift_degree=2, lifting_points=[]):
    """Compute values for intermediate variables.

    Keyword arguments:
        x0  -- start values for independent variable and variables at current lifting points.
        num_coeffs  -- number of coefficients of the polynomial
        lift_degree -- degree of component functions
        lifting_points -- pre-determined lifting given by list of binary decisions
    """
    max_degree = num_coeffs - 1
    x_dim = 2
    x_init = [x0[:x_dim]]

    if (max_degree <= 0):
        raise ValueError("ERROR: degree of polynomial has to be greater than 0")

    num_lifts = int(np.round(np.log(max_degree) / np.log(lift_degree), 2)) + 1

    for i in range(1, num_lifts):
        try:
            lift_curr = lifting_points[i]
        except Exception:
            lift_curr = 0

        if (lift_curr):
            # take point from input
            start_ind = i * x_dim
            stop_ind = (i + 1) * x_dim
            x_new = x0[start_ind:stop_ind]
        else:
            # compute from previous state
            x_new = complex_utils.complex_exponent(x_init[i - 1], lift_degree)

        x_init += [x_new]

    return cs.vertcat(*x_init)

