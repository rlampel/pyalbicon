import casadi as cs
from . import base_class


class Problem(base_class.BVP):
    lamb = 1
    x_dim = 4

    def __init__(self):
        base_class.BVP()

    def get_ode(self):
        lamb = self.lamb
        # Declare model variables
        x = cs.MX.sym('x', self.x_dim)
        t = cs.MX.sym('t', 1)

        # Model equations
        x0dot = cs.sin(x[1])
        x1dot = x[2]
        x2dot = -x[3] / lamb
        x3dot = (x[0] - 1) * cs.cos(x[1]) - x[2] * 1 / cs.cos(x[1])
        x3dot += lamb * x[3] * cs.tan(x[1])
        x3dot /= lamb
        xdot = cs.vertcat(x0dot, x1dot, x2dot, x3dot)

        # Objective term
        ode = {'x': x, 'ode': xdot, 't': t}
        return ode

    def get_boundary_fct(self):
        x_start = cs.MX.sym('xs', self.x_dim)
        x_end = cs.MX.sym('xe', self.x_dim)

        # declare boundary function
        bvp1 = x_start[0]
        bvp2 = x_start[2]
        bvp3 = x_end[0]
        bvp4 = x_end[2]
        bvs = cs.vertcat(bvp1, bvp2, bvp3, bvp4)
        R = cs.Function('R', [x_start, x_end], [bvs], ['xs', 'xe'], ['bvp'])
        return R

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([2., 1.65, 1., 1.])
        init["s_dim"] = self.x_dim
        return init

    def get_grid_details(self):
        min_t = 0
        max_t = 1
        num_lifts = 10
        return min_t, max_t, num_lifts

