import numpy as np
import casadi as cs


def initialize(init_vals, grid, ode, init_type="auto", bounds=None):
    """Initialize the intermediate variables for an ode.

    Keyword arguments:
        init_vals   --  dict containing the independent variable and dimension
        grid    -- dict containing the discretization and lifting points
        ode     --  casadi function
        init_type   -- how to initialize the intermediate variables
    """
    if (init_type == "auto"):
        s_init = initialize_auto(init_vals, grid, ode)
    elif (init_type == "lin"):
        s_init = initialize_lin(init_vals, grid)
    else:
        s_init = initialize_random(init_vals, grid, bounds)
    return s_init


def initialize_lin(init_vals, grid):
    """Initialize all values as the starting points or interpolate linearly
    between given start and end point.

    Keyword arguments:
        init_vals   --  dict containing the independent variable and dimension
        grid    -- dict containing the discretization and lifting points
    """
    s_start = init_vals["s_start"]
    s_end = init_vals.get("s_end", s_start)
    time_points = grid["time"]

    start_time = time_points[0]
    end_time = time_points[-1]   # end point
    incline = (s_end - s_start) / (end_time - start_time)
    s_init = cs.DM([])

    for t in time_points:
        curr_val = s_start + incline * (t - start_time)
        s_init = cs.vertcat(s_init, curr_val)

    return s_init


def initialize_random(init_vals, grid, bounds):
    """Initialize all intermediate variables randomly.

    Keyword arguments:
        init_vals   --  dict containing the independent variable and dimension
        grid    -- dict containing the discretization and lifting points
    """
    s_dim = init_vals["s_dim"]
    time_points = grid["time"]
    s_list = []

    if (bounds is None):
        up_bounds = cs.DM([1] * s_dim)
        low_bounds = cs.DM([-1] * s_dim)
    else:
        up_bounds = bounds["upper"]
        low_bounds = bounds["lower"]

    for t in range(len(time_points)):
        rand_vals = np.random.uniform(low_bounds, up_bounds)
        s_list += list(rand_vals.flatten())

    print(s_list)
    return cs.DM(s_list)


def initialize_auto(init_vals, grid, ode):
    """Initialize all values via FSInit

    Keyword arguments:
        init_vals   --  dict containing the independent variable and dimension
        grid    -- dict containing the discretization and lifting points
        ode     -- casadi function
    """
    curr_s = init_vals["s_start"]
    grid["lift"] = grid.get("lift", [0 for el in grid["time"]])
    init_vals["sol"] = init_vals.get("sol", curr_s)
    return compute_all_states(init_vals, grid, ode)


def compute_all_states(init, grid, ode):
    """Initialize all values that are not contained in the current lifting via FSInit.

    Keyword arguments:
        init   --  dict containing the independent variable and dimension
        grid    -- dict containing the discretization and lifting points
        ode     -- casadi function
    """
    s_dim = init["s_dim"]
    sol = init["sol"]

    time_points = grid["time"]
    lifting_points = grid["lift"]

    s_temp = sol

    curr_lift_point = 0
    all_states = cs.DM([])

    curr_s = s_temp[curr_lift_point * s_dim:(curr_lift_point + 1) * s_dim]
    all_states = cs.vertcat(all_states, curr_s)
    opts = {"show_eval_warnings": False, "common_options": {"show_eval_warnings": False}}

    for j in range(len(time_points) - 1):
        if (lifting_points[j + 1] == 1):
            curr_lift_point += 1
            curr_s = s_temp[curr_lift_point * s_dim:(curr_lift_point + 1) * s_dim]
        else:
            Ik = cs.integrator('F', 'rk', ode, time_points[j], time_points[j + 1], opts)
            curr_s = Ik(x0=curr_s)["xf"]

        all_states = cs.vertcat(all_states, curr_s)
    return all_states


def convert_lifting(input_lift, time_points, tol=1e-8):
    """Converts a lifting given by the time points into a list of binary decisions for every
    lifting point.

    Keyword arguments:
        input_lift  -- list containing the lifting points as time points
        time_points -- list of all possible lifting points as time points
    """
    input_lift = np.asarray(input_lift)
    time_points = np.asarray(time_points)

    # Compare each time_point against all input_lift values
    mask = np.any(np.isclose(time_points[:, None], input_lift[None, :], atol=tol), axis=1)
    return mask.astype(int).tolist()


def select_states(s_init, s_dim, lifting_points):
    """Return a vector containing only the points in the current lifting out of all intermediate
    variables.

    Keyword arguments:
        s_init  -- casadi DM containing all intermediate variables
        s_dim   -- dimension of one intermediate variable
        lifting_points  -- list with binary entries for every lifting point
    """
    # Precompute indices of selected blocks
    indices = []
    for i, lp in enumerate(lifting_points):
        if lp or i == 0:
            indices.extend(range(i * s_dim, (i + 1) * s_dim))

    # Use CasADi indexing once
    return s_init[indices]

