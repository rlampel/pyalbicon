import casadi as cs
import matplotlib.pyplot as plt
import utils.get_problem as get_p
import utils.create_bvp as create_bvp
import utils.initialization as initialization
import utils.fast_newton as newton
import utils.lifting as lifting
import utils.heatmap as heatmap
import utils.heatmap_iter as heatmap_iter
import timeit
import os


# settings
log_results = True  # write results into log file
plot_contr_region = False  # plot the local contraction as a heatmap
plot_iter_region = False  # plot the number of required iterations as a heatmap
plot_auto_lift = True  # plot the current lifting for every step of auto_lifted_newton
plot_results = True  # plot the convergence comparison for the different algorithms
plot_delay = 0.25          # how long to show each newton iteration for repeated lifting
result_delay = 2        # how long to show the convergence plots
verbose = False        # print out all Newton iterations

# file where the results will be saved
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, "results/benchmark.log")

if (log_results):
    f = open(filename, "w")
    header = "BVP | lambda | starting point | def. | Alg. 3 | rep. Alg. 3 | Alg. 5 + rep. Alg. 3"
    f.write(header + "\n")
    f.close()

for p in range(19, 34):
    curr_name = "T" + str(p)

    # get problem details
    problem = get_p.get_problem(p)
    ode = problem.get_ode()
    R = problem.get_boundary_fct()
    min_t, max_t, num_lifts = problem.get_grid_details()

    init = problem.get_init()
    s_dim = init["s_dim"]

    # define time grid
    grid = {}
    time_points = [min_t + (max_t - min_t) * i / num_lifts for i in range(num_lifts + 1)]
    grid["time"] = time_points

    # plot local contraction as a heatmap
    if (plot_contr_region and s_dim == 2):
        plot_dim = 21       # resolution of the heatmap
        xlb, xub = -5, 5     # boundaries of the heatmap
        lb, ub = 0, 2        # limit contraction to values in [0,2]

        a = heatmap.plot_heatmap_default(ode, R, xlb, xub, plot_dim)
        g = heatmap.plot_heatmap_graph(ode, R, xlb, xub, plot_dim)
        h = heatmap.plot_heatmap_auto_const(problem, xlb, xub, plot_dim)

        fig, axes = plt.subplots(1, 3, figsize=(15, 6), dpi=100, sharey=True)
        im1 = axes[0].imshow(a, vmin=lb, vmax=ub)
        axes[0].set_xlabel(r"initial $x_0$", size=16)
        axes[0].set_ylabel(r"initial $x_1$", size=16)
        axes[0].set_xticks([0, (plot_dim - 1) / 2, plot_dim - 1])
        axes[0].set_xticklabels([xlb, (xlb + xub) / 2, xub], fontsize=14)
        axes[0].set_yticks([0, (plot_dim - 1) / 2, plot_dim - 1])
        axes[0].set_yticklabels([xub, (xlb + xub) / 2, xlb], fontsize=14)

        im2 = axes[1].imshow(g, vmin=lb, vmax=ub)
        axes[1].set_xlabel(r"initial $x_0$", size=16)
        axes[1].set_xticks([0, (plot_dim - 1) / 2, plot_dim - 1])
        axes[1].set_xticklabels([xlb, (xlb + xub) / 2, xub], fontsize=14)

        im3 = axes[2].imshow(h, vmin=lb, vmax=ub)
        axes[2].set_xlabel(r"initial $x_0$", size=16)
        axes[2].set_xticks([0, (plot_dim - 1) / 2, plot_dim - 1])
        axes[2].set_xticklabels([xlb, (xlb + xub) / 2, xub], fontsize=14)

        # add colorbar
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([axes[-1].get_position().x1 + 0.05,
                                axes[-1].get_position().y0,
                                0.05,
                                axes[-1].get_position().height])
        colorbar = fig.colorbar(im1, cax=cbar_ax)
        colorbar.set_ticks([0, 0.5, 1, 1.5, 2])
        colorbar.set_label(label=r"contraction $h$", fontsize=16)
        colorbar.set_ticklabels([0, 0.5, 1, 1.5, r"$\geq$2"], fontsize=14)
        plt.show()
    if (plot_iter_region and s_dim == 2):
        plot_dim = 21       # resolution of the heatmap
        xlb, xub = -5, 5     # boundaries of the heatmap
        lb, ub = 0, 20        # limit contraction to values in [0,2]
        cmap = "Grays"

        a = heatmap_iter.plot_heatmap_default(ode, R, xlb, xub, plot_dim)
        g = heatmap_iter.plot_heatmap_graph(ode, R, xlb, xub, plot_dim)
        h = heatmap_iter.plot_heatmap_auto_const(problem, xlb, xub, plot_dim)

        fig, axes = plt.subplots(1, 3, figsize=(15, 6), dpi=100, sharey=True)
        im1 = axes[0].imshow(a, vmin=lb, vmax=ub, cmap=cmap)
        axes[0].set_xlabel(r"initial $x_0$", size=16)
        axes[0].set_ylabel(r"initial $x_1$", size=16)
        axes[0].set_xticks([0, (plot_dim - 1) / 2, plot_dim - 1])
        axes[0].set_xticklabels([xlb, (xlb + xub) / 2, xub], fontsize=14)
        axes[0].set_yticks([0, (plot_dim - 1) / 2, plot_dim - 1])
        axes[0].set_yticklabels([xub, (xlb + xub) / 2, xlb], fontsize=14)

        im2 = axes[1].imshow(g, vmin=lb, vmax=ub, cmap=cmap)
        axes[1].set_xlabel(r"initial $x_0$", size=16)
        axes[1].set_xticks([0, (plot_dim - 1) / 2, plot_dim - 1])
        axes[1].set_xticklabels([xlb, (xlb + xub) / 2, xub], fontsize=14)

        im3 = axes[2].imshow(h, vmin=lb, vmax=ub, cmap=cmap)
        axes[2].set_xlabel(r"initial $x_0$", size=16)
        axes[2].set_xticks([0, (plot_dim - 1) / 2, plot_dim - 1])
        axes[2].set_xticklabels([xlb, (xlb + xub) / 2, xub], fontsize=14)

        # add colorbar
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([axes[-1].get_position().x1 + 0.05,
                                axes[-1].get_position().y0,
                                0.05,
                                axes[-1].get_position().height])
        colorbar = fig.colorbar(im1, cax=cbar_ax)
        colorbar.set_ticks([0, 5, 10, 15, 20])
        colorbar.set_label(label=r"number of iterations", fontsize=16)
        colorbar.set_ticklabels([0, 5, 10, 15, r"$\geq$20"], fontsize=14)
        plt.show()

    # create unlifted version
    if (verbose):
        print("----" * 10)
        print("UNLIFTED:")
    grid["lift"] = [0 for i in range(len(time_points))]

    start_time = timeit.default_timer()
    B_def = create_bvp.create_bvp(ode, R, grid, s_dim)
    _, func_arr = newton.newton(B_def, init["s_start"],
                                opts={"verbose": verbose})
    elapsed = timeit.default_timer() - start_time
    print("-" * 20 + "\nBVP" + str(p))
    print("No lifting took", elapsed)
    default_conv = [float(el) for el in func_arr]

    # lift at every point to compute all possible steps
    s_init = initialization.initialize(init, grid, ode)
    # s_init = initialization.initialize_lin(init, grid)
    grid["lift"] = [1 for i in range(len(time_points))]

    start_time = timeit.default_timer()
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
    # replace by condensed newton
    from utils import condensing
    if lift_init.shape[0] > s_dim:
        func_list, RF = create_bvp.create_condensing_bvp(ode, R, grid, s_dim)
        _, func_arr = condensing.condensed_newton(lift_init[:s_dim], lift_init[s_dim:],
                                                  func_list, RF)
    else:
        B_graph_lift = create_bvp.create_bvp(ode, R, grid, s_dim)
        _, func_arr = newton.newton(B_graph_lift, lift_init,
                                    opts={"verbose": verbose})
    elapsed = timeit.default_timer() - start_time
    print("Graph Lifting took ", elapsed)

    graph_conv = [float(el) for el in func_arr]

    # best lifting for every iteration
    if (verbose):
        print("----" * 10)
        print("AUTOMATIC LIFTING:")
    start_time = timeit.default_timer()
    func_arr = newton.auto_lifted_newton(problem,
                                         opts={"verbose": verbose,
                                               "plot": plot_auto_lift,
                                               "plot_delay": plot_delay})
    elapsed = timeit.default_timer() - start_time
    print("Auto Lifting took ", elapsed)
    auto_conv = [float(el) for el in func_arr]

    # heuristic lifting
    # constant initialization for all variables
    start_vals = initialization.initialize_lin(init, grid)
    start_time = timeit.default_timer()
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
    elapsed = timeit.default_timer() - start_time
    print("Heuristic Auto Lifting took ", elapsed)
    heur_auto_conv = [float(el) for el in func_arr]

    if (log_results):
        f = open(filename, "a")
        def_iter = str(len(default_conv) - 1)
        fs_iter = str(len(graph_conv) - 1)
        auto_iter = str(len(auto_conv) - 1)
        heur_auto_iter = str(len(heur_auto_conv) - 1)
        start_log = str(init["s_start"])
        table_list = curr_name + " & " + str(problem.lamb) + " & $" + start_log + "$ & $"
        table_list += def_iter + "$ & $" + fs_iter + "$ & $" + auto_iter
        table_list += "$ & $" + heur_auto_iter + "$ \\\\ \n"
        f.write(table_list)
        f.close()

    if (plot_results):
        plt.plot([i for i in range(len(default_conv))], default_conv, label="no lifting")
        plt.plot([i for i in range(len(graph_conv))], graph_conv, label="graph lifting",
                 linestyle="-.")
        plt.plot([i for i in range(len(auto_conv))], auto_conv, label="auto lifting",
                 linestyle="--")
        plt.plot([i for i in range(len(heur_auto_conv))], heur_auto_conv, label="heuristic + auto",
                 linestyle="--")

        plt.legend()
        plt.yscale("log")
        plt.title("BVP T" + str(p))
        plt.pause(result_delay)
        plt.close()

