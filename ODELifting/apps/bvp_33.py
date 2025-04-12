import casadi as cs
from . import base_class


class Problem(base_class.BVP):
    lamb = 1
    x_dim = 6

    def __init__(self):
        base_class.BVP()

    def get_ode(self):
        lamb = self.lamb
        # Declare model variables
        x = cs.MX.sym('x', self.x_dim)
        t = cs.MX.sym('t', 1)

        # Model equations
        x0dot = x[1]
        x1dot = 1 / lamb * x[0] * x[3] - x[2] * x[1]
        x2dot = x[3]
        x3dot = x[4]
        x4dot = x[5]
        x5dot = 1 / lamb * (-x[2] * x[5] - x[0] * x[1])
        xdot = cs.vertcat(x0dot, x1dot, x2dot, x3dot, x4dot, x5dot)

        # Objective term
        ode = {'x': x, 'ode': xdot, 't': t}
        return ode

    def get_boundary_fct(self):
        x_start = cs.MX.sym('xs', self.x_dim)
        x_end = cs.MX.sym('xe', self.x_dim)

        # declare boundary function
        bvp1 = x_start[0] + 1
        bvp2 = x_start[2]
        bvp3 = x_start[3]
        bvp4 = x_end[0] - 1
        bvp5 = x_end[2]
        bvp6 = x_end[3]

        bvs = cs.vertcat(bvp1, bvp2, bvp3, bvp4, bvp5, bvp6)
        R = cs.Function('R', [x_start, x_end], [bvs], ['xs', 'xe'], ['bvp'])
        return R

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([50, 30, 50, 40, 30, 30])
        init["s_dim"] = self.x_dim
        return init

    def get_grid_details(self):
        min_t = 0
        max_t = 1
        num_lifts = 10
        return min_t, max_t, num_lifts

