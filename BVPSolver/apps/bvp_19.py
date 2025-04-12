import casadi as cs
import numpy as np
from . import base_class


class Problem(base_class.BVP):
    lamb = 0.05
    x_dim = 2

    def __init__(self):
        base_class.BVP()

    def get_ode(self):
        lamb = self.lamb
        # Declare model variables
        x = cs.MX.sym('x', self.x_dim)
        t = cs.MX.sym('t', 1)

        # Model equations
        x0dot = x[1]
        x1dot = -cs.exp(x[0]) * x[1] + np.pi / 2 * cs.sin(np.pi * t / 2) * cs.exp(2 * x[0])
        x1dot /= lamb
        xdot = cs.vertcat(x0dot, x1dot)

        # Objective term
        ode = {'x': x, 'ode': xdot, 't': t}
        return ode

    def get_boundary_fct(self):
        x_start = cs.MX.sym('xs', self.x_dim)
        x_end = cs.MX.sym('xe', self.x_dim)

        # declare boundary function
        bvp0 = x_start[0]
        bvp1 = x_end[0]
        bvs = cs.vertcat(bvp0, bvp1)
        R = cs.Function('R', [x_start, x_end], [bvs], ['xs', 'xe'], ['bvp'])
        return R

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([-2., 4.2])
        init["s_dim"] = self.x_dim
        return init

    def get_grid_details(self):
        min_t = 0
        max_t = 1
        num_lifts = 10
        return min_t, max_t, num_lifts

