import matplotlib.pyplot as plt
import casadi as cs
import numpy as np
import utils.newton as newton
import utils.create_poly as create_poly
import utils.greedy_lift as greedy_lift
import os


log_results = False  # write results to a file
lift_degree = 2  # degree of the component functions
plot_delay = 0.01  # how many seconds to show the results
TOL = 1.e-12  # final residual tolerance

if (log_results):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "poly_data/poly_conv.dat")

    f = open(filename, "w")
    log_header = "max degree | def | greedy | enum | greedy mult | enum mult \n"
    f.write(log_header)
    f.close()

start = cs.DM([5., 0.])

num_reps = 100

np.random.seed(42)

# plot the correspondence between initial contraction and the total number of iterations
iter_list = []
contr_list = []

for poly_dim in range(12, 18, 2):
    print("-" * 20)
    avg_default = 0
    avg_greedy = 0
    avg_enum = 0
    avg_greedy_ml = 0
    avg_enum_ml = 0

    avg_default_size = 0
    avg_greedy_size = 0
    avg_enum_size = 0
    avg_greedy_ml_size = 0
    avg_enum_ml_size = 0
    for i in range(num_reps):
        neg_roots = -np.random.rand(poly_dim - 2) * 1
        pos_root = np.random.rand(1) * 1
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
        avg_default += len(plot_vals)
        avg_default_size += sol.shape[0] / 2

        contr_list += [plot_vals[1] / plot_vals[0]]
        iter_list += [len(plot_vals) - 1]

        sol, plot_vals = newton.newton(G1, s1)
        plt.plot([i for i in range(1, len(plot_vals) + 1)], plot_vals, label="greedy (b)")
        avg_greedy += len(plot_vals)
        avg_greedy_size += sol.shape[0] / 2

        contr_list += [plot_vals[1] / plot_vals[0]]
        iter_list += [len(plot_vals) - 1]

        sol, plot_vals = newton.newton(G2, s2)
        plt.plot([i for i in range(1, len(plot_vals) + 1)], plot_vals, label="enum (b)")
        avg_enum += len(plot_vals)
        avg_enum_size += sol.shape[0] / 2

        sol, plot_vals = newton.newton(G1m, s1m)
        plt.plot([i for i in range(1, len(plot_vals) + 1)], plot_vals,
                 label="greedy (a)", linestyle="--")
        avg_greedy_ml += len(plot_vals)
        avg_greedy_ml_size += sol.shape[0] / 2
        final_val = float(plot_vals[-1])
        if (final_val >= TOL or np.isnan(final_val)):
            avg_greedy_ml = np.inf

        sol, plot_vals = newton.newton(G2m, s2m)
        plt.plot([i for i in range(1, len(plot_vals) + 1)], plot_vals,
                 label="enum (a)", linestyle="--")
        avg_enum_ml += len(plot_vals)
        avg_enum_ml_size += sol.shape[0] / 2
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
    text_out += f"{avg_default / num_reps} & "
    text_out += f"{avg_greedy / num_reps} & "
    text_out += f"{avg_enum / num_reps} & "
    text_out += f"{avg_greedy_ml / num_reps} & "
    text_out += f"{avg_enum_ml / num_reps} \\\\ \n "

    print(text_out)
    if (log_results):
        f = open(filename, "a")
        f.write(text_out)
        f.close()

    plt.scatter(contr_list, iter_list)
    plt.show()
    iter_list = []
    contr_list = []
