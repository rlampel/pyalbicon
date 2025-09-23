import casadi as cs
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils.newton as newton
import matplotlib as mpl
from concurrent.futures import ProcessPoolExecutor

# globals for workers
_global_R_lifted = None
_global_R = None
# fonts
mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)

plot_dim = 201  # resolution of the plot


def fs_init(x_start):
    x_out = x_start
    x_out = cs.vertcat(x_start[0], x_start[0]**2, x_start[1])
    return x_out


def create_rosenbrock(lift=True):
    # create original variables
    X0 = cs.MX.sym('X0')
    X1 = cs.MX.sym('X1')
    S = cs.DM([])
    RHS = cs.DM([])

    # create lifting conditions
    if lift:
        X_sqr = cs.MX.sym("X_sqr")
        lift_cond = X0**2 - X_sqr
        S = X_sqr
        RHS = cs.vertcat(RHS, lift_cond)
    else:
        X_sqr = X0**2

    # create derivative of Rosenbrock function
    x0_der = 400 * (X0 * X_sqr - X0 * X1)
    x0_der += 2 * X0 - 2
    x1_der = -200 * (X_sqr - X1)
    RHS = cs.vertcat(RHS, x0_der, x1_der)

    S = cs.vertcat(X0, S, X1)
    G = cs.Function('R', [S], [RHS])
    return G


def best_initial_lift(G, x_start, opts={}):
    """Computes the best lifting for the Rosenbrock function. Since there is
    only one possible lifting point, this reduces to a single comparison.

    Keyword arguments:
        G       -- casadi function for which Newton's method is performed
        x_start -- start point of Newton's method
    """
    # compute the normal lifted Newton step
    lift = False
    x_next = x_start

    x_next, _ = newton.newton(G, x_next, {"max_iter": 1})
    x_auto = fs_init(cs.DM([x_next[0], x_next[2]]))
    if (cs.norm_2(G(x_next)) <= cs.norm_2(G(x_auto))):
        lift = True
    return lift


def _init_worker(lifted):
    """Initialize R in each worker (avoid pickling CasADi object)."""
    global _global_R, _global_R_lifted
    _global_R = create_rosenbrock(False)
    _global_R_lifted = create_rosenbrock(True)


def _compute_point(args):
    row, col, x1, x2, plot_dim, auto_lifting = args
    curr_val = cs.DM([x1, x2])
    if auto_lifting:
        temp_val = fs_init(curr_val)
        lifted = best_initial_lift(_global_R_lifted, temp_val)
        if lifted:
            _, func_arr = newton.newton(_global_R_lifted, temp_val)
        else:
            _init_worker(lifted)
            _, func_arr = newton.newton(_global_R, curr_val)
    else:
        _, func_arr = newton.newton(_global_R, curr_val)
    num_iter = len(func_arr) - 1
    return (row, col, num_iter)


def create_heatmap_parallel(xb, yb, plot_dim, auto_lifting=False, max_workers=None):
    """Parallelized function that creates a heatmap of the required iterations."""
    xlb, xub = xb
    ylb, yub = yb
    plot_vals_x = np.linspace(xlb, xub, plot_dim)
    plot_vals_y = np.linspace(ylb, yub, plot_dim)
    plot_matrix = np.zeros((plot_dim, plot_dim))

    tasks = []
    for row in range(plot_dim):
        x2 = plot_vals_y[-1 - row]  # keep same row ordering
        for col in range(plot_dim):
            x1 = plot_vals_x[col]
            tasks.append((row, col, x1, x2, plot_dim, auto_lifting))

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(auto_lifting,)
    ) as executor:
        for row, col, num_iter in tqdm(executor.map(_compute_point, tasks), total=len(tasks)):
            plot_matrix[row, col] = num_iter

    return plot_matrix


xlb, xub = -2, 2
ylb, yub = -1, 3
xb = [xlb, xub]
yb = [ylb, yub]
plot_default = create_heatmap_parallel(xb, yb, plot_dim)
plot_lift = create_heatmap_parallel(xb, yb, plot_dim, True)
max_val_def = np.max(plot_default)
max_val_lift = np.max(plot_lift)
max_val = np.max([max_val_def, max_val_lift])

fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=100, sharey=True)
im1 = axes[0].imshow(plot_default, vmin=0, vmax=max_val, cmap=mpl.colormaps["Greys"])
axes[0].set_xlabel(r"$\bar{x}_0$", size=16)
axes[0].set_ylabel(r"$\bar{x}_1$", size=16)
axes[0].set_xticks([0, (plot_dim - 1) / 2, plot_dim - 1])
axes[0].set_xticklabels([xlb, (xlb + xub) / 2, xub], fontsize=14)
axes[0].set_yticks([0, (plot_dim - 1) / 2, plot_dim - 1])
axes[0].set_yticklabels([yub, (ylb + yub) / 2, ylb], fontsize=14)
axes[0].set_title("default", fontsize=22)

im2 = axes[1].imshow(plot_lift, vmin=0, vmax=max_val, cmap=mpl.colormaps["Greys"])
axes[1].set_xlabel(r"$\bar{x}_0$", size=16)
axes[1].set_xticks([0, (plot_dim - 1) / 2, plot_dim - 1])
axes[1].set_xticklabels([xlb, (xlb + xub) / 2, xub], fontsize=14)
axes[1].set_title("lifted", fontsize=22)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([axes[-1].get_position().x1 + 0.05,
                        axes[-1].get_position().y0,
                        0.05,
                        axes[-1].get_position().height])
colorbar = fig.colorbar(im1, cax=cbar_ax)
for t in colorbar.ax.get_yticklabels():
    t.set_fontsize(14)
colorbar.set_label(label=r"number of iterations", fontsize=16)
tick_distance = np.round(max_val / 10)
if (tick_distance <= 0):
    ticks = [i for i in range(int(max_val) + 1)]
else:
    ticks = [np.round(tick_distance * i) for i in range(10)]
    ticks = [el for el in ticks if el <= max_val]
colorbar.set_ticks(ticks)

plt.show()
