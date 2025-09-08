import casadi as cs
import numpy as np
from . import newton, create_bvp, initialization, lifting
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


# globals for workers
_global_ode = None
_global_R = None
_global_problem = None


def compute_contraction(B, s0, s1):
    """Computes the residual contraction for a given function and two subsequent Newton iterates.

    Keyword arguments:
        B   -- casadi function
        s0  -- start point
        s1  -- value after one Newton iteration
    """
    return cs.norm_2(B(s1)) / cs.norm_2(B(s0))


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
    x_next, _ = newton.newton(B, x_start, {"max_iter": 1, "verbose": False})
    contr = compute_contraction(B, x_start, x_next)

    if np.isnan(contr):
        contr = np.inf
    return (plot_dim - 1 - i, j, contr)


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
        for row, col, contr in tqdm(executor.map(_compute_point_default, tasks), total=len(tasks)):
            plot_matrix[row, col] = contr

    return plot_matrix


def _compute_point_graph(args):
    i, j, x1, x2, time_points, plot_dim, x_dim = args
    x_start = cs.DM([x1, x2])

    B = create_bvp.create_bvp(_global_ode, _global_R, {"time": list(time_points)}, x_dim)
    x_next, _ = newton.newton(B, x_start, {"max_iter": 1, "verbose": False})
    contr = compute_contraction(B, x_start, x_next)

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
                                         time_points, s_next, time_points, 2)
    graph_lift = list(graph_lift)
    lifting_points = initialization.convert_lifting(graph_lift, time_points)
    s_best = initialization.select_states(s_init, 2, lifting_points)
    B = create_bvp.create_bvp(_global_ode, _global_R,
                              {"time": time_points, "lift": lifting_points}, x_dim)
    s_next, _ = newton.newton(B, s_best, {"max_iter": 1})
    # compute DT
    contr = compute_contraction(B, s_best, s_next)

    if np.isnan(contr):
        contr = np.inf
    return (plot_dim - 1 - i, j, contr)


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
        for row, col, contr in tqdm(executor.map(_compute_point_graph, tasks), total=len(tasks)):
            plot_matrix[row, col] = contr

    return plot_matrix


def _compute_point_graph_init(args):
    i, j, x1, x2, time_points, plot_dim, x_dim = args
    x_start = cs.DM([x1, x2])

    start_vals = initialization.initialize({"s_start": x_start, "s_dim": x_dim},
                                           {"time": time_points}, _global_ode, "lin")

    graph_lift = lifting.best_graph_lift(_global_ode, _global_R, time_points, start_vals,
                                         time_points, x_dim)
    graph_lift = list(graph_lift)

    # compute contraction
    lifting_points = initialization.convert_lifting(graph_lift, time_points)
    s_best = initialization.select_states(start_vals, 2, lifting_points)
    opts = {"verbose": False, "max_iter": 1, "plot": False}
    func_arr = newton.auto_lifted_newton(_global_problem, lifting_points, s_best, opts)
    if (len(func_arr) == 1):
        contr = np.inf
    else:
        contr = func_arr[1] / func_arr[0]
    return (plot_dim - 1 - i, j, contr)


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
        for row, col, contr in tqdm(executor.map(_compute_point_graph_init, tasks), total=len(tasks)):
            plot_matrix[row, col] = contr
    return plot_matrix


def plot_heatmap_auto_random_graph(problem, xlb, xub, plot_dim, random_scale,
                                   seed=42):
    """Returns a numpy array with plot_dim rows and columns that contains the local residual
    contractions of the optimal lifting of the BVP, defined via the problem class, for values within
    the interval [xlb, xub]^2. The intermediate values are determined using random perturbations.

    Keyword arguments:
        problem     -- instance of the problem class for a bvp
        xlb, xub    -- lower and upper bounds of interval
        plot_dim    -- number of sample points within [xlb, xub]
        random_scale    -- sample range for random numbers [-random_scale, random_scale]
    """
    ode = problem.get_ode()
    R = problem.get_boundary_fct()
    x_dim = 2
    sample_range = np.linspace(xlb, xub, plot_dim)
    plot_matrix = np.zeros((plot_dim, plot_dim))

    time_points = list(np.linspace(0., 1., 11))

    for i in tqdm(range(plot_dim)):
        x2 = sample_range[i]
        for j in range(plot_dim):
            x1 = sample_range[j]
            x_start = np.array([x1, x2])  # compute all intermediate variables
            start_vals = initialization.initialize({"s_start": x_start, "s_dim": x_dim},
                                                   {"time": time_points}, ode, "auto")

            # add random noise
            np.random.seed(seed)
            noise = np.random.uniform(-1, 1, len(time_points) * x_dim) * random_scale

            # remove noise of original variable
            noise[:x_dim] = [0] * x_dim
            noise = cs.DM(list(noise))
            start_vals = start_vals + noise

            graph_lift = lifting.best_graph_lift(ode, R, time_points, start_vals,
                                                 time_points, x_dim)
            graph_lift = list(graph_lift)

            # compute contraction
            lifting_points = initialization.convert_lifting(graph_lift, time_points)
            s_best = initialization.select_states(start_vals, 2, lifting_points)
            opts = {"verbose": False, "max_iter": 1, "plot": False}
            func_arr = newton.auto_lifted_newton(problem, lifting_points, s_best, opts)
            if (len(func_arr) == 1):
                contr = np.inf
            else:
                contr = func_arr[1] / func_arr[0]

            plot_matrix[plot_dim - 1 - i, j] = contr
    return plot_matrix


