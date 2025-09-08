import casadi as cs
import numpy as np
from . import base_class


class Problem(base_class.BVP):
    lamb = 0.05

    def __init__(self, seed=42, x_dim=20):
        base_class.BVP.__init__(self)
        self.seed = seed
        self.x_dim = x_dim

    def get_ode(self):
        # Declare model variables
        x = cs.MX.sym('x', self.x_dim)
        t = cs.MX.sym('t', 1)

        # Model equations
        xdot = cs.DM([])
        np.random.seed(self.seed)
        for i in range(self.x_dim):
            b = cs.DM(np.random.rand(self.x_dim))
            b = cs.DM([1.] * self.x_dim) - 2 * b
            c = 1 - 2 * float(np.random.rand(1))
            xdot_i = b.T @ x + c
            xdot = cs.vertcat(xdot, xdot_i)

        # Objective term
        ode = {'x': x, 'ode': xdot, 't': t}
        return ode

    def get_boundary_fct(self):
        x_start = cs.MX.sym('xs', self.x_dim)
        x_end = cs.MX.sym('xe', self.x_dim)

        # declare boundary function
        bvs = x_end - 0.5 * x_start  # cs.DM([1] * self.x_dim)
        R = cs.Function('R', [x_start, x_end], [bvs], ['xs', 'xe'], ['bvp'])
        return R

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([0.] * self.x_dim)
        init["s_dim"] = self.x_dim
        return init

    def get_grid_details(self):
        min_t = 0
        max_t = 10
        num_lifts = 20
        return min_t, max_t, num_lifts

