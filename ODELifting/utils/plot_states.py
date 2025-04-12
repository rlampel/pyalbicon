import casadi as cs
import numpy as np

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2
mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)


def plot_segmented(fig, init, grid, ode, starting_ind=0):
    """Plot an ODE with given shooting points.

    Keyword arguments:
        fig -- matplotlib figure in which to plot
        init -- dict containing intermediate variables and dimension
        grid -- dict containing time and lifting points
        ode  -- casadi function
    """
    s_dim = init["s_dim"]
    sol = init["sol"]

    ax = fig.add_subplot(1, 1, 1)

    time_points = grid["time"]
    lifting_points = grid["lift"]

    # convert to numpy array
    sol = np.array(sol).flatten()

    colors = ["black", "red", "#777", "#955", "brown", "purple"]
    styles = ["-", "--", ":", "-."]

    curr_lift_point = 0
    curr_s = sol[curr_lift_point * s_dim:(curr_lift_point + 1) * s_dim]
    plot_list = [np.array(curr_s).flatten()]

    for j in range(len(time_points) - 1):
        curr_init = {}
        curr_init["s"] = curr_s
        Ik = cs.integrator('F', 'rk', ode, time_points[j], time_points[j + 1])
        curr_s = Ik(x0=curr_s)["xf"]
        plot_list += [np.array(curr_s).flatten()]

        if (lifting_points[j + 1] == 1):
            for d in range(s_dim):
                ax.plot(time_points[starting_ind:j + 2],
                        [el[d] for el in plot_list],
                        color=colors[d % len(colors)],
                        linestyle=styles[d % 4])

            starting_ind = j + 1
            curr_lift_point += 1
            curr_s = sol[curr_lift_point * s_dim:(curr_lift_point + 1) * s_dim]
            plot_list = [np.array(curr_s).flatten()]

    for d in range(s_dim):
        ax.plot(time_points[starting_ind:len(time_points)],
                [el[d] for el in plot_list],
                color=colors[d % len(colors)],
                linestyle=styles[d % 4])

    legend_labels = []
    for var_index in range(s_dim):
        label = r"$x_{" + str(var_index) + "}$"
        legend_labels += [label]

    yl, yu = ax.get_ylim()
    ax.vlines([time_points[k] for k in range(len(time_points)) if lifting_points[k] == 1],
              yl, yu, color="gray", linestyles="dashed", alpha=0.5)

    ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    ax.set_xlabel("Time t", fontsize=16)
    ax.set_ylabel("Component value", fontsize=16)
    return ax

