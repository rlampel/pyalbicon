import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils.create_poly as create_poly
import utils.initialize as initialize
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


x_interv = [-1, 1]  # range of real values
y_interv = [-1, 1]  # range of imaginary values
res = 50  # number of sampling points in interval [0,1]

coeffs = [-2] + [0] * 15 + [1]


def trust_region(G, x, dx, start_mu=1, TOL=1.e-6):
    lam_min = 1.e-16
    mu = start_mu
    if (np.isnan(mu)):
        mu = np.inf
        lam = 1
    else:
        lam = np.min([1, mu])
    new_lam = 0

    while (True):
        # print("search with lambda: ", lam)
        if (np.abs(lam) < lam_min):
            return cs.DM_inf(x.shape[0])

        x_next = x + lam * dx
        # compute monitoring quantities
        theta = cs.norm_2(G(x_next)) / cs.norm_2(G(x))
        if (np.isnan(theta)):
            theta = np.inf

        mu = 0.5 * lam**2 * cs.norm_2(G(x)) / cs.norm_2(G(x_next) - (1 - lam) * G(x))
        mu = float(mu)
        if (np.isnan(mu)):
            mu = np.inf

        if (theta >= 1 - lam / 4):
            new_lam = np.min(np.array([mu, 0.5 * lam]).flatten())
            lam = new_lam
        else:
            break

    return x + lam * dx


def newton(G, start, TOL=1.e-6, max_iter=50, line_search=False):
    counter = 0
    x = start

    y = cs.MX.sym('y', x.shape[0])
    DG = cs.Function('J', [y], [cs.jacobian(G(y), y)], ['x_in'], ['J'])
    curr_norm = float(cs.norm_2(G(x)))
    if (np.isnan(curr_norm)):
        return cs.DM_inf(2), max_iter

    while (counter < max_iter and curr_norm >= TOL):
        dx = -cs.solve(DG(x), G(x))
        if (line_search):
            x_new = trust_region(G, x, dx)
        else:
            x_new = x + dx

        x = x_new
        curr_norm = float(cs.norm_2(G(x)))
        if (curr_norm == cs.inf or np.isnan(curr_norm)):
            return cs.DM_inf(2), max_iter
        counter += 1
    return x[:2], counter


def plot_convergence(coeffs, x_interv, y_interv, res, lift_degree=1,
                     line_search=False):
    if (lift_degree == 1):
        f = create_poly.create_poly(coeffs)
    else:
        f = create_poly.create_lifted_poly(coeffs, lift_degree)

    plot_dimx = int(np.round(np.abs(x_interv[1] - x_interv[0]) * res) + 1)
    plot_dimy = int(np.round(np.abs(y_interv[1] - y_interv[0]) * res) + 1)

    replot = np.zeros((plot_dimy, plot_dimx))
    implot = np.zeros((plot_dimy, plot_dimx))
    stepplot = np.zeros((plot_dimy, plot_dimx))
    plot_pointsx = np.linspace(x_interv[0], x_interv[1], plot_dimx)
    plot_pointsy = np.linspace(y_interv[0], x_interv[1], plot_dimy)
    for i in tqdm(range(plot_dimx)):
        re = plot_pointsx[i]
        for j in range(plot_dimy):
            im = plot_pointsy[j]
            start = cs.DM([re, im])
            if (lift_degree > 1):
                start = initialize.initialize_auto(start, len(coeffs), lift_degree)
            root, steps = newton(f, start, line_search=line_search)
            replot[plot_dimy - 1 - j, i] = float(root[0])
            implot[plot_dimy - 1 - j, i] = float(root[1])
            stepplot[plot_dimy - 1 - j, i] = steps
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


replot, implot, stepplot = plot_convergence(coeffs, x_interv, y_interv, res,
                                            lift_degree=2, line_search=False)
roots = np.polynomial.polynomial.polyroots(coeffs)
plot = color_grading(roots, replot, implot)

fig, ax = plt.subplots(1, 1, figsize=(15, 10), dpi=100, sharey=True)

cmap = plt.cm.viridis.resampled(len(roots) + 1)
im = ax.imshow(plot, cmap=cmap)

# ax.imshow(-stepplot, alpha=0.25, cmap="gray")
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
ax2.set_xticks([0, dimx - 1], [x_interv[0], x_interv[1]], fontsize=14)
ax2.set_yticks([0, dimy - 1], [y_interv[1], y_interv[0]], fontsize=14)
ax2.set_xlabel(r"real part", size=16)
ax2.set_ylabel(r"imaginary part", size=16)
plt.show()

