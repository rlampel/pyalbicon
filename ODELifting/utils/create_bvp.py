import casadi as cs
import numpy as np


def get_next_time_point(curr_time, time_points):
    """Get the next time point from an array that comes after the current time.

    Keyword arguments:
        curr_time   -- current time point
        time_points -- list of all time points
    """
    for t in time_points:
        if (curr_time < t and not np.isclose(curr_time, t)):
            return t
    return -1


def create_bvp(ode, R, grid, x_dim):
    """Create a boundary value function from given ODE, boundary function R, time and lifting points

    Keyword arguments:
        ode     -- casadi function that describes the ODE
        R       -- boundary function
        grid    -- dict that contains the discretization and the lifting points
        x_dim   -- dimension of the states
    """
    time_points = grid["time"]  # possible lifting points
    N = len(time_points) - 1
    # binary decision whether to lift at a time point
    lifting_points = grid.get("lift", [0] * (N + 1))
    RHS = cs.DM([])
    Sk = cs.MX.sym('S_0', x_dim)

    S = Sk  # lifted input
    Sk_temp = Sk
    for k in range(N):
        opts = {"show_eval_warnings": False, "common_options": {"show_eval_warnings": False}}
        t_curr, t_next = time_points[k], time_points[k + 1]
        # integrate to next possible lifting point
        Ik = cs.integrator('F', 'rk', ode, t_curr, t_next, opts)
        Sk_end = Ik(x0=Sk_temp)["xf"]
        if lifting_points[k + 1]:
            # introduce new lifted variable
            Sk_temp = cs.MX.sym('S_' + str(k + 1), x_dim)
            S = cs.vertcat(S, Sk_temp)
            RHS = cs.vertcat(RHS, Sk_end - Sk_temp)
        else:
            Sk_temp = Sk_end

    RHS = cs.vertcat(RHS, R(Sk, Sk_temp))
    G = cs.Function('G', [S], [RHS])
    return G
