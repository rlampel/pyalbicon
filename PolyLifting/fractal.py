import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils.create_poly as create_poly
import utils.initialize as initialize
from matplotlib import rc
from concurrent.futures import ProcessPoolExecutor

# globals for workers
_global_coeffs = None
_global_lift_degree = None
_global_line_search = None
_global_f = None
# font settings
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


# settings
x_interv = [-1, 1]  # range of real values
y_interv = [-1, 1]  # range of imaginary values
res = 100  # number of sampling points in interval [0,1]
lift_degree = 1  # choose 2 for the lifted function and 1 for the default function
line_search = False  # choose False for Figure 3 and True for Figure 4

coeffs = [-2] + [0] * 15 + [1]


def newton(G, start, TOL=1.e-6, max_iter=50, line_search=False):
    counter = 0
    x = start

    curr_norm = float(cs.norm_2(G(x)))
    if (np.isnan(curr_norm)):
        return cs.DM_inf(2), max_iter

    newton_step = cs.rootfinder('newton_step', 'newton', G,
                                {'line_search': line_search, 'max_iter': 1, 'error_on_fail': False})
    while (counter < max_iter and curr_norm >= TOL):
        x = newton_step(x)
        curr_norm = float(cs.norm_2(G(x)))
        if (curr_norm == cs.inf or np.isnan(curr_norm)):
            return cs.DM_inf(2), max_iter
        counter += 1
    return x[:2], counter


def _init_worker(coeffs, lift_degree, line_search):
    """Initialize CasADi function f in each worker."""
    global _global_coeffs, _global_lift_degree, _global_line_search, _global_f
    _global_coeffs = coeffs
    _global_lift_degree = lift_degree
    _global_line_search = line_search
    if lift_degree == 1:
        _global_f = create_poly.create_poly(coeffs)
    else:
        _global_f = create_poly.create_lifted_poly(coeffs, lift_degree)


def _compute_point(args):
    i, j, re, im, plot_dimy, lift_degree = args
    start = cs.DM([re, im])
    if lift_degree > 1:
        start = initialize.initialize_auto(start, len(_global_coeffs), lift_degree)

    root, steps = newton(_global_f, start, line_search=_global_line_search)

    return (plot_dimy - 1 - j, i, float(root[0]), float(root[1]), steps)


def plot_convergence_parallel(coeffs, x_interv, y_interv, res, lift_degree=1,
                              line_search=False, max_workers=None):
    """Parallelized version of plot_convergence."""
    plot_dimx = int(np.round(np.abs(x_interv[1] - x_interv[0]) * res) + 1)
    plot_dimy = int(np.round(np.abs(y_interv[1] - y_interv[0]) * res) + 1)

    replot = np.zeros((plot_dimy, plot_dimx))
    implot = np.zeros((plot_dimy, plot_dimx))
    stepplot = np.zeros((plot_dimy, plot_dimx))

    plot_pointsx = np.linspace(x_interv[0], x_interv[1], plot_dimx)
    plot_pointsy = np.linspace(y_interv[0], y_interv[1], plot_dimy)

    tasks = []
    for i, re in enumerate(plot_pointsx):
        for j, im in enumerate(plot_pointsy):
            tasks.append((i, j, re, im, plot_dimy, lift_degree))

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(coeffs, lift_degree, line_search)
    ) as executor:
        for row, col, re_val, im_val, steps in tqdm(
            executor.map(_compute_point, tasks), total=len(tasks)
        ):
            replot[row, col] = re_val
            implot[row, col] = im_val
            stepplot[row, col] = steps

    return replot, implot, stepplot


def color_grading(roots, replot, implot, TOL=1.e-5):
    plot_dimx, plot_dimy = replot.shape
    color_plot = np.zeros((plot_dimx, plot_dimy))
    for i in range(plot_dimx):
        for j in range(plot_dimy):
            x = replot[i, j]
            y = implot[i, j]
            if (np.isnan(x) or np.isnan(y)):
                color_plot[i, j] = 0
            else:
                val = x + y * 1j
                for k in range(len(roots)):
                    r = roots[k]
                    if (np.abs(val - r) < TOL):
                        color_plot[i, j] = k + 1
    return color_plot


replot, implot, stepplot = plot_convergence_parallel(coeffs, x_interv,
                                                     y_interv, res,
                                                     lift_degree=lift_degree,
                                                     line_search=line_search)
roots = np.polynomial.polynomial.polyroots(coeffs)
plot = color_grading(roots, replot, implot)

fig, ax = plt.subplots(1, 1, figsize=(15, 10), dpi=100, sharey=True)

cmap = plt.cm.viridis.resampled(len(roots) + 1)
im = ax.imshow(plot, cmap=cmap)

cax = fig.add_axes([ax.get_position().x1 + 0.05,
                    ax.get_position().y0,
                    0.05,
                    ax.get_position().height])
colorbar = fig.colorbar(im, cax=cax)
colorbar.set_ticks(ticks=[k for k in range(len(roots) + 1)])
colorbar.set_ticklabels([np.inf] + [r"root $" + str(k) + "$" for k in range(len(roots))],
                        fontsize=14)

dimx, dimy = stepplot.shape

ax.set_xlabel(r"real part", size=16)
ax.set_ylabel(r"imaginary part", size=16)
ax.set_xticks([0, dimx - 1], [x_interv[0], x_interv[1]], fontsize=14)
ax.set_yticks([0, dimy - 1], [y_interv[1], y_interv[0]], fontsize=14)
plt.show()

# plot number of iterations
fig2, ax2 = plt.subplots(1, 1, figsize=(15, 10), dpi=100)
im2 = ax2.imshow(stepplot, cmap="Grays")
cax = fig2.add_axes([ax2.get_position().x1 + 0.05,
                    ax2.get_position().y0,
                    0.05,
                    ax2.get_position().height])
colorbar = fig2.colorbar(im2, cax=cax)
colorbar.set_ticks(ticks=[10 * k for k in range(1, 6)])
colorbar.set_label("Number of iterations", size=16)
ax2.set_xticks([0, dimx - 1], [x_interv[0], x_interv[1]], fontsize=14)
ax2.set_yticks([0, dimy - 1], [y_interv[1], y_interv[0]], fontsize=14)
ax2.set_xlabel(r"real part", size=16)
ax2.set_ylabel(r"imaginary part", size=16)
plt.show()

