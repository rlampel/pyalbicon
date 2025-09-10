import casadi as cs
import numpy as np
from . import newton, create_bvp, initialization, lifting
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


# globals for workers
_global_ode = None
_global_R = None
_global_problem = None
_global_max_iter = 20


def _init_worker(ode, R):
    global _global_ode, _global_R
    _global_ode = ode
    _global_R = R


def _init_problem(problem):
    global _global_problem, _global_ode, _global_R
    _global_problem = problem
    _global_ode = problem.get_ode()
    _global_R = problem.get_boundary_fct()


def _compute_point_default(args):
    i, j, x1, x2, time_points, plot_dim, x_dim = args
    x_start = cs.DM([x1, x2])

    B = create_bvp.create_bvp(_global_ode, _global_R, {"time": list(time_points)}, x_dim)
    _, func_arr = newton.newton(B, x_start, {"max_iter": _global_max_iter, "verbose": False})

    final_res = func_arr[-1]
    num_iter = len(func_arr)

    if np.isnan(final_res) or final_res == np.inf:
        num_iter = _global_max_iter
    return (plot_dim - 1 - i, j, num_iter)


def plot_heatmap_default(ode, R, xlb, xub, plot_dim, max_workers=None):
    """Returns a numpy array with plot_dim rows and columns that contains the local residual
    contractions of the BVP, defined via ode and R, for values within the interval [xlb, xub]^2.

    Keyword arguments:
        ode -- casadi function
        R   -- casadi function
        xlb, xub    -- lower and upper bounds of interval
        plot_dim    -- number of sample points within [xlb, xub]
    """
    x_dim = 2
    sample_range = np.linspace(xlb, xub, plot_dim)
    plot_matrix = np.zeros((plot_dim, plot_dim))
    time_points = np.linspace(0, 1, 11)

    tasks = []
    for i in range(plot_dim):
        x2 = sample_range[i]
        for j in range(plot_dim):
            x1 = sample_range[j]
            tasks.append((i, j, x1, x2, time_points, plot_dim, x_dim))

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker, initargs=(ode, R)) as executor:
        for row, col, num_iter in tqdm(executor.map(_compute_point_default, tasks), total=len(tasks)):
            plot_matrix[row, col] = num_iter

    return plot_matrix


def _compute_point_graph(args):
    i, j, x1, x2, time_points, plot_dim, x_dim = args
    x_start = cs.DM([x1, x2])

    # compute all intermediate variables
    s_init = initialization.initialize({"s_start": x_start, "s_dim": x_dim},
                                       {"time": time_points}, _global_ode, "auto")
    # lift at every point
    lifting_points = [1] * len(time_points)
    # perform one Newton step
    B = create_bvp.create_bvp(_global_ode, _global_R,
                              {"time": time_points, "lift": lifting_points}, x_dim)
    s_next, _ = newton.newton(B, s_init, {"max_iter": 1})

    graph_lift = lifting.best_graph_lift(_global_ode, _global_R,
                                         time_points, s_next, time_points, 2,
                                         parallel=False)
    graph_lift = list(graph_lift)
    lifting_points = initialization.convert_lifting(graph_lift, time_points)
    s_best = initialization.select_states(s_init, 2, lifting_points)
    B = create_bvp.create_bvp(_global_ode, _global_R,
                              {"time": time_points, "lift": lifting_points}, x_dim)
    _, func_arr = newton.newton(B, s_best, {"max_iter": _global_max_iter})

    final_res = func_arr[-1]
    num_iter = len(func_arr)

    if np.isnan(final_res) or final_res == np.inf:
        num_iter = _global_max_iter
    return (plot_dim - 1 - i, j, num_iter)


def plot_heatmap_graph(ode, R, xlb, xub, plot_dim, max_workers=None):
    """Returns a numpy array with plot_dim rows and columns that contains the local residual
    contractions of the optimal lifting of the BVP, defined via ode and R, for values within
    the interval [xlb, xub]^2.

    Keyword arguments:
        ode -- casadi function
        R   -- casadi function
        xlb, xub    -- lower and upper bounds of interval
        plot_dim    -- number of sample points within [xlb, xub]
    """
    x_dim = 2
    sample_range = np.linspace(xlb, xub, plot_dim)
    plot_matrix = np.zeros((plot_dim, plot_dim))
    time_points = list(np.linspace(0., 1., 11))

    tasks = []
    for i in range(plot_dim):
        x2 = sample_range[i]
        for j in range(plot_dim):
            x1 = sample_range[j]
            tasks.append((i, j, x1, x2, time_points, plot_dim, x_dim))

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker, initargs=(ode, R)) as executor:
        for row, col, num_iter in tqdm(executor.map(_compute_point_graph, tasks), total=len(tasks)):
            plot_matrix[row, col] = num_iter

    return plot_matrix


def _compute_point_graph_init(args):
    i, j, x1, x2, time_points, plot_dim, x_dim = args
    x_start = cs.DM([x1, x2])

    start_vals = initialization.initialize({"s_start": x_start, "s_dim": x_dim},
                                           {"time": time_points}, _global_ode, "lin")

    graph_lift = lifting.best_graph_lift(_global_ode, _global_R, time_points, start_vals,
                                         time_points, x_dim, parallel=False)
    graph_lift = list(graph_lift)

    # compute contraction
    lifting_points = initialization.convert_lifting(graph_lift, time_points)
    s_best = initialization.select_states(start_vals, 2, lifting_points)
    opts = {"verbose": False, "max_iter": _global_max_iter, "plot": False}
    func_arr = newton.auto_lifted_newton(_global_problem, lifting_points, s_best, opts)

    final_res = func_arr[-1]
    num_iter = len(func_arr)

    if np.isnan(final_res) or final_res == np.inf:
        num_iter = _global_max_iter

    return (plot_dim - 1 - i, j, num_iter)


def plot_heatmap_auto_const(problem, xlb, xub, plot_dim, max_workers=None):
    """Returns a numpy array with plot_dim rows and columns that contains the local residual
    contractions of the optimal lifting of the BVP, defined via the problem class, for values within
    the interval [xlb, xub]^2. The intermediate values are determined using constant initialization.

    Keyword arguments:
        problem     -- instance of the problem class for a bvp
        xlb, xub    -- lower and upper bounds of interval
        plot_dim    -- number of sample points within [xlb, xub]
        random_scale    -- sample range for random numbers [-random_scale, random_scale]
    """
    x_dim = 2
    sample_range = np.linspace(xlb, xub, plot_dim)
    plot_matrix = np.zeros((plot_dim, plot_dim))
    time_points = list(np.linspace(0., 1., 11))

    tasks = []
    for i in range(plot_dim):
        x2 = sample_range[i]
        for j in range(plot_dim):
            x1 = sample_range[j]
            tasks.append((i, j, x1, x2, time_points, plot_dim, x_dim))

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_problem, initargs=[problem]) as executor:
        for row, col, num_iter in tqdm(executor.map(_compute_point_graph_init, tasks), total=len(tasks)):
            plot_matrix[row, col] = num_iter
    return plot_matrix

