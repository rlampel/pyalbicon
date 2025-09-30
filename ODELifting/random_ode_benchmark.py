import matplotlib.pyplot as plt
import numpy as np
import utils.create_bvp as create_bvp
import utils.initialization as initialization
import utils.newton as newton
import utils.lifting as lifting
import apps.bvp_42 as random_problem
import os


# settings
log_results = False  # write results into log file
delete_log = False  # erase all previous entries in the log file
plot_auto_lift = False  # plot the current lifting for every step of auto_lifted_newton
plot_results = True  # plot the convergence comparison for the different algorithms
plot_delay = 1.5          # how long to show each newton iteration for repeated lifting
result_delay = 2        # how long to show the convergence plots
verbose = False        # print out all Newton iterations


# all dimensions for which the BVP are being solved
dims = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
num_reps = 5  # number of runs per dimension
counter = 0


# file where the results will be saved
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, "results/random_ode_benchmark.log")

if (log_results and delete_log):
    f = open(filename, "w")
    header = "dim | def. | Alg. 3 | rep. Alg. 3 | Alg. 5 + rep. Alg. 3"
    f.write(header + "\n")
    f.close()

for curr_dim in dims:
    print(f"Dimension {curr_dim}")
    def_iter = 0
    fs_iter = 0
    auto_iter = 0
    heur_auto_iter = 0

    def_contr = 0
    graph_contr = 0
    heur_auto_contr = 0

    for rep in range(num_reps):
        counter += 1
        # get problem details
        problem = random_problem.Problem(seed=counter, x_dim=curr_dim)
        ode = problem.get_ode()
        R = problem.get_boundary_fct()
        min_t, max_t, num_lifts = problem.get_grid_details()

        init = problem.get_init()
        s_dim = init["s_dim"]

        # define time grid
        grid = {}
        time_points = [min_t + (max_t - min_t) * i / num_lifts for i in range(num_lifts + 1)]
        grid["time"] = time_points

        # create unlifted version
        if (verbose):
            print("----" * 10)
            print("UNLIFTED:")
        grid["lift"] = [0 for i in range(len(time_points))]
        B_def = create_bvp.create_bvp(ode, R, grid, s_dim)
        # capped at 20 iterations to save time, it does not converge with more iterations
        _, func_arr = newton.newton(B_def, init["s_start"],
                                    opts={"verbose": verbose, "max_iter": 20})
        default_conv = [float(el) for el in func_arr]
        def_contr += default_conv[1] / default_conv[0]

        # lift at every point to compute all possible steps
        s_init = initialization.initialize(init, grid, ode)
        grid["lift"] = [1 for i in range(len(time_points))]
        B_lift_all = create_bvp.create_bvp(ode, R, grid, s_dim)
        first_iter, first_norms = newton.newton(B_lift_all, s_init,
                                                opts={"verbose": False, "max_iter": 1})

        # graph-based lifting
        if (verbose):
            print("----" * 10)
            print("GRAPH LIFTING:")
        graph_lift = lifting.best_graph_lift(ode, R, time_points, first_iter, time_points,
                                             s_dim, verbose=False)
        grid["lift"] = initialization.convert_lifting(graph_lift, time_points)
        # the first step has already been computed by the lifting algorithm, start at step 2
        lift_init = initialization.select_states(first_iter, s_dim, grid["lift"])
        B_graph_lift = create_bvp.create_bvp(ode, R, grid, s_dim)
        _, func_arr = newton.newton(B_graph_lift, lift_init,
                                    opts={"verbose": verbose})
        graph_conv = [float(first_norms[0])] + [float(el) for el in func_arr]
        graph_contr += graph_conv[1] / graph_conv[0]

        # heuristic lifting
        # constant initialization for all variables
        start_vals = initialization.initialize_lin(init, grid)
        heur_lift = lifting.best_graph_lift(ode, R, time_points, start_vals, time_points, s_dim)
        grid["lift"] = initialization.convert_lifting(heur_lift, time_points)
        s_best = initialization.select_states(start_vals, s_dim, grid["lift"])

        # graph based heuristic lifting
        if (verbose):
            print("----" * 10)
            print("HEURISTIC + AUTO:")
        func_arr = newton.auto_lifted_newton(problem, grid["lift"], s_best,
                                             {"verbose": verbose,
                                              "plot": plot_auto_lift,
                                              "plot_delay": plot_delay})
        heur_auto_conv = [float(el) for el in func_arr]
        heur_auto_contr += heur_auto_conv[1] / heur_auto_conv[0]

        if (plot_results):
            plt.plot([i for i in range(len(default_conv))], default_conv, label="no lifting")
            plt.plot([i for i in range(len(graph_conv))], graph_conv, label="graph lifting",
                     linestyle="-.")
            plt.plot([i for i in range(len(heur_auto_conv))], heur_auto_conv,
                     label="heuristic + auto",
                     linestyle="--")

            plt.legend()
            plt.yscale("log")
            plt.title("dimension " + str(curr_dim))
            plt.xlabel(r"Iteration $k$")
            plt.ylabel(r"Residual $\|G(\mathbf{s}^{(k)})\|$")
            plt.pause(result_delay)
            plt.close()

        def_iter += len(default_conv) - 1
        fs_iter += len(graph_conv) - 1
        heur_auto_iter += len(heur_auto_conv) - 1

    if (log_results):
        # transform into the right format with exponents
        avg_def_contr = np.format_float_scientific(def_contr / num_reps, precision=3)
        avg_graph_contr = np.format_float_scientific(graph_contr / num_reps, precision=3)
        avg_heur_contr = np.format_float_scientific(heur_auto_contr / num_reps, precision=3)
        f = open(filename, "a")
        table_list = "dim. " + str(curr_dim)
        table_list += "$ & $" + str(def_iter / num_reps) + f"({avg_def_contr})"
        table_list += "$ & $" + str(fs_iter / num_reps) + f"({avg_graph_contr})"
        table_list += "$ & $" + str(heur_auto_iter / num_reps) + f"({avg_heur_contr})"
        table_list += "$ \\\\ \n"
        f.write(table_list)
        f.close()
