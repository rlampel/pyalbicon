import matplotlib.pyplot as plt
import casadi as cs
import numpy as np
import utils.newton as newton
import utils.create_poly as create_poly
import utils.greedy_lift as greedy_lift
import os


log_results = False  # write results to a file
log_contraction = False  # log the initial contractions and number of iterations
lift_degree = 2  # degree of the component functions
plot_delay = 0.01  # how many seconds to show the results
TOL = 1.e-8  # final residual tolerance

if (log_results):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "poly_data/poly_conv.dat")

    f = open(filename, "w")
    log_header = "max degree | def | greedy | enum | greedy mult | enum mult \n"
    f.write(log_header)
    f.close()


num_reps = 100

np.random.seed(42)

for poly_dim in range(5, 18, 2):
    # plot the correspondence between initial contraction and the total number of iterations
    print("-" * 20)
    avg_default = 0
    avg_greedy = 0
    avg_enum = 0
    avg_greedy_ml = 0
    avg_enum_ml = 0

    avg_default_contr = 0
    avg_greedy_contr = 0
    avg_enum_contr = 0
    avg_greedy_ml_contr = 0
    avg_enum_ml_contr = 0

    for i in range(num_reps):
        start = cs.DM([5, 0.])

        neg_roots = -np.random.rand(poly_dim - 2) * 1.
        pos_root = np.random.rand(1) * 1.
        roots = np.append(neg_roots, pos_root)

        coeffs = create_poly.create_coeffs(roots)

        # compute lifting and first Newton step simultaneously
        F = create_poly.create_poly(coeffs)
        G1, s1 = greedy_lift.greedy_start(coeffs, start, lift_degree)
        G2, s2 = greedy_lift.enumerate_start(coeffs, start, lift_degree)
        G1m, s1m = greedy_lift.greedy_start(coeffs, start, lift_degree, lift_type="multilin")
        G2m, s2m = greedy_lift.enumerate_start(coeffs, start, lift_degree, lift_type="multilin")

        sol, plot_vals = newton.newton(F, start)
        plt.plot([i for i in range(1, len(plot_vals))], plot_vals[1:], label="default")
        avg_default += len(plot_vals) - 1
        init_res = plot_vals[0]  # initial residual at start point
        avg_default_contr += plot_vals[1] / init_res

        sol, plot_vals = newton.newton(G1, s1)
        plt.plot([i for i in range(1, len(plot_vals) + 1)], plot_vals, label="greedy (b)")
        avg_greedy += len(plot_vals)
        avg_greedy_contr += plot_vals[0] / init_res
        final_val = float(plot_vals[-1])
        if (final_val >= TOL or np.isnan(final_val)):
            avg_greedy = np.inf

        sol, plot_vals = newton.newton(G2, s2)
        plt.plot([i for i in range(1, len(plot_vals) + 1)], plot_vals, label="enum (b)")
        avg_enum += len(plot_vals)
        avg_enum_contr += plot_vals[0] / init_res
        final_val = float(plot_vals[-1])
        if (final_val >= TOL or np.isnan(final_val)):
            avg_enum = np.inf

        sol, plot_vals = newton.newton(G1m, s1m)
        plt.plot([i for i in range(1, len(plot_vals) + 1)], plot_vals,
                 label="greedy (a)", linestyle="--")
        avg_greedy_ml += len(plot_vals)
        avg_greedy_ml_contr += plot_vals[0] / init_res
        final_val = float(plot_vals[-1])
        if (final_val >= TOL or np.isnan(final_val)):
            avg_greedy_ml = np.inf

        sol, plot_vals = newton.newton(G2m, s2m)
        plt.plot([i for i in range(1, len(plot_vals) + 1)], plot_vals,
                 label="enum (a)", linestyle="--")
        avg_enum_ml += len(plot_vals)
        avg_enum_ml_contr += plot_vals[0] / init_res
        final_val = float(plot_vals[-1])
        if (final_val >= TOL or np.isnan(final_val)):
            avg_enum_ml = np.inf

        plt.legend(loc="upper right")
        plt.xlabel("iteration")
        plt.ylabel("residual error")
        plt.yscale("log")
        plt.title("highest degree: " + str(poly_dim - 1))
        plt.pause(plot_delay)
        plt.clf()

    text_out = f"{poly_dim - 1} & "
    text_out += f"{avg_default / num_reps} ({avg_default_contr / num_reps:.3f}) & "
    text_out += f"{avg_greedy / num_reps} ({avg_greedy_contr / num_reps:.3f}) & "
    text_out += f"{avg_enum / num_reps} ({avg_enum_contr / num_reps:.3f}) & "
    text_out += f"{avg_greedy_ml / num_reps} ({avg_greedy_ml_contr / num_reps:.3f}) & "
    text_out += f"{avg_enum_ml / num_reps} ({avg_enum_ml_contr / num_reps:.3f}) \\\\ \n "

    print(text_out)
    if (log_results):
        f = open(filename, "a")
        f.write(text_out)
        f.close()

