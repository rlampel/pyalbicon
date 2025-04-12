import numpy as np
import casadi as cs
from . import complex_utils


def create_coeffs(roots):
    """Compute the coefficients of a polynomial with given roots.

    Keyword arguments:
        roots   -- list of roots
    """
    coeffs = np.array([roots[0], 1])

    for i in range(1, len(roots)):
        neg_root = -roots[i]
        coeffs1 = np.append([0], coeffs)
        coeffs2 = np.append(coeffs * neg_root, [0])
        coeffs = coeffs1 + coeffs2
    return coeffs


def create_poly(coeffs):
    """Create a polynomial with given coefficients

    Keyword arguments:
        coeffs  -- list of coefficients
    """
    num_coeffs = len(coeffs)
    X = cs.MX.sym("X", 2)
    Y = 0
    for i in range(num_coeffs):
        Y += coeffs[i] * complex_utils.complex_exponent(X, i)

    P = cs.Function("F", [X], [Y], ["X"], ["Y"])
    return P


def create_lifted_poly(coeffs, lift_degree=2, lifting_points=[], lift_type=""):
    """Create a lifted polynomial with given coefficients and degree of component functions.

    Keyword arguments:
        coeffs  -- list of coefficients
        lift_degree -- degree of component functions
        lifting_points -- pre-determined lifting given by list of binary decisions
        lift_type -- "multilin" for mulitilinear lifting, else lifting with fractional exponents
    """
    num_coeffs = len(coeffs)
    max_degree = num_coeffs - 1
    Y = cs.DM([])

    if (max_degree <= 0):
        raise ValueError("ERROR: degree of polynomial has to be greater than 0")

    num_lifts = int(np.round(np.log(max_degree) / np.log(lift_degree), 2)) + 1

    if (len(lifting_points) == 0):
        # lift at all points
        lifting_points = [1] * num_lifts
    elif (len(lifting_points) != num_lifts):
        raise ValueError("wrong number of lifting points supplied")

    X0 = cs.MX.sym("X0", 2)
    S = [X0]

    Xk = X0
    for i in range(1, num_lifts):
        if (lifting_points[i]):
            # add lifting point
            Xk_old = Xk
            Xk = cs.MX.sym("X" + str(i), 2)
            S += [Xk]
            Y = cs.vertcat(Y, complex_utils.complex_exponent(Xk_old, lift_degree) - Xk)
        else:
            # skip lifting point
            Xk = complex_utils.complex_exponent(Xk, lift_degree)

    var_exp = 1
    skipped_ind = 0
    lift_ind = 0
    var_ind = 0
    last_comp = 0

    if (lift_type == "multilin"):
        # variant 1: including multilinear terms
        for j in range(num_coeffs):
            var_ind = -1
            curr_exp = j
            curr_mult = cs.DM([coeffs[j], 0])
            # walk backwards through all lifted variables
            for k in range(1, num_lifts + 1):
                if (coeffs[j] == 0):
                    break
                curr_var_exp = lift_degree**(num_lifts - k)
                if (k == num_lifts):
                    lift_var = complex_utils.complex_exponent(S[var_ind], curr_exp)
                    curr_mult = complex_utils.complex_mult(curr_mult, lift_var)
                    break

                rem_exp = curr_exp // curr_var_exp
                if (lifting_points[-k]):
                    if (rem_exp > 0):
                        lift_var = complex_utils.complex_exponent(S[var_ind], rem_exp)
                        curr_mult = complex_utils.complex_mult(curr_mult, lift_var)
                        curr_exp -= rem_exp * curr_var_exp
                    var_ind -= 1
                if (curr_exp == 0):
                    break
            last_comp += curr_mult
    else:
        # variant 2: no multilinear terms
        for j in range(num_coeffs):
            curr_exp = j
            curr_mult = coeffs[j]
            if (curr_exp >= lift_degree * var_exp):
                var_exp *= lift_degree
                skipped_ind += 1
                lift_ind += 1
                if (lifting_points[lift_ind]):
                    var_ind += 1
                    skipped_ind = 0
            exp_mult = complex_utils.complex_exponent(S[var_ind],
                                                      curr_exp * (1 + skipped_ind) / var_exp)
            last_comp += curr_mult * exp_mult

    Y = cs.vertcat(Y, last_comp)
    S = cs.vertcat(*S)
    P = cs.Function("F", [S], [Y], ["X"], ["Y"])
    return P

