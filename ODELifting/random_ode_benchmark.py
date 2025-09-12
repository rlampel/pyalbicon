import matplotlib.pyplot as plt
import utils.create_bvp as create_bvp
import utils.initialization as initialization
import utils.newton as newton
import utils.lifting as lifting
import apps.bvp_42 as random_problem
import os


# settings
log_results = True  # write results into log file
delete_log = False  # erase all previous entries in the log file
plot_auto_lift = True  # plot the current lifting for every step of auto_lifted_newton
plot_results = True  # plot the convergence comparison for the different algorithms
plot_delay = 0.25          # how long to show each newton iteration for repeated lifting
result_delay = 2        # how long to show the convergence plots
verbose = True        # print out all Newton iterations

dims = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
num_reps = 5
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
        _, func_arr = newton.newton(B_def, init["s_start"],
                                    opts={"verbose": verbose, "max_iter": 20})
        default_conv = [float(el) for el in func_arr]

        # lift at every point to compute all possible steps
        s_init = initialization.initialize(init, grid, ode)
        grid["lift"] = [1 for i in range(len(time_points))]
        B_lift_all = create_bvp.create_bvp(ode, R, grid, s_dim)
        first_iter, _ = newton.newton(B_lift_all, s_init,
                                      opts={"verbose": False, "max_iter": 1})

        # graph-based lifting
        if (verbose):
            print("----" * 10)
            print("GRAPH LIFTING:")
        graph_lift = lifting.best_graph_lift(ode, R, time_points, first_iter, time_points,
                                             s_dim, verbose=False)
        grid["lift"] = initialization.convert_lifting(graph_lift, time_points)
        lift_init = initialization.select_states(s_init, s_dim, grid["lift"])
        B_graph_lift = create_bvp.create_bvp(ode, R, grid, s_dim)
        _, func_arr = newton.newton(B_graph_lift, lift_init,
                                    opts={"verbose": verbose})
        graph_conv = [float(el) for el in func_arr]

        # best lifting for every iteration
        if (verbose):
            print("----" * 10)
            print("AUTOMATIC LIFTING:")
        func_arr = newton.auto_lifted_newton(problem,
                                             opts={"verbose": verbose,
                                                   "plot": plot_auto_lift,
                                                   "plot_delay": plot_delay})
        auto_conv = [float(el) for el in func_arr]

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

        if (log_results):
            f = open(filename, "a")
            def_iter = str(len(default_conv) - 1)
            fs_iter = str(len(graph_conv) - 1)
            auto_iter = str(len(auto_conv) - 1)
            heur_auto_iter = str(len(heur_auto_conv) - 1)
            start_log = str(init["s_start"])
            table_list = "dim. " + str(curr_dim)
            table_list += "$ & $" + def_iter + "$ & $" + fs_iter + "$ & $" + auto_iter
            table_list += "$ & $" + heur_auto_iter + "$ \\\\ \n"
            f.write(table_list)
            f.close()

        if (plot_results):
            plt.plot([i for i in range(len(default_conv))], default_conv, label="no lifting")
            plt.plot([i for i in range(len(graph_conv))], graph_conv, label="graph lifting",
                     linestyle="-.")
            plt.plot([i for i in range(len(auto_conv))], auto_conv, label="auto lifting",
                     linestyle="--")
            plt.plot([i for i in range(len(heur_auto_conv))], heur_auto_conv,
                     label="heuristic + auto",
                     linestyle="--")

            plt.legend()
            plt.yscale("log")
            plt.title("dimension " + str(curr_dim))
            plt.pause(result_delay)
            plt.close()

