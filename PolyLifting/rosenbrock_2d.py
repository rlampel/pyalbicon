import casadi as cs
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils.newton as newton
import matplotlib as mpl
mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)


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


def create_heatmap(xb, yb, plot_dim, lifted=False):
    xlb, xub = xb[0], xb[1]
    ylb, yub = yb[0], yb[1]
    plot_vals_x = np.linspace(xlb, xub, plot_dim)
    plot_vals_y = np.linspace(ylb, yub, plot_dim)
    plot_matrix = np.zeros((plot_dim, plot_dim))
    if (lifted):
        R = create_rosenbrock(True)
    else:
        R = create_rosenbrock(False)

    for row in tqdm(range(plot_dim)):
        x2 = plot_vals_y[-1 - row]
        for col in range(plot_dim):
            x1 = plot_vals_x[col]
            curr_val = cs.DM([x1, x2])
            if (lifted):
                curr_val = fs_init(curr_val)
            _, func_arr = newton.newton(R, curr_val)
            num_iter = len(func_arr) - 1

            plot_matrix[row, col] = num_iter
    return plot_matrix


xlb, xub = -2, 2
ylb, yub = -1, 3
xb = [xlb, xub]
yb = [ylb, yub]
plot_dim = 51
plot_default = create_heatmap(xb, yb, plot_dim)
plot_lift = create_heatmap(xb, yb, plot_dim, True)
max_val_def = np.max(plot_default)
max_val_lift = np.max(plot_lift)
max_val = np.max([max_val_def, max_val_lift])
print("max_val: ", max_val_def, max_val_lift, max_val)

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
