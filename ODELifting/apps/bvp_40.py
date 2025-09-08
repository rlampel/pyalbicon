import casadi as cs
from . import base_class


# two body problem
class Problem(base_class.BVP):
    x_dim = 12
    lamb = 1
    c1, c2 = 0.4, 0.2

    def __init__(self):
        base_class.BVP()

    def get_ode(self):
        c1, c2 = self.c1, self.c2
        # Declare model variables
        x = cs.MX.sym('x', 2)
        lam0 = cs.MX.sym('lam0', 2)
        y = cs.MX.sym('y', 2)
        lam1 = cs.MX.sym('lam1', 2)
        z = cs.MX.sym('z', 2)
        lam2 = cs.MX.sym('lam2', 2)
        t = cs.MX.sym('t', 1)
        t_final = 12

        # Model equations
        u1 = 0
        u2 = 1

        # singular curve
        x1, x2 = z[0], z[1]

        numerator = c1**3 * x1**3 - c2**3 * x2**3 + c1**3 * x1**2 * x2 - c2**3 * x1 * x2**2
        numerator += 2 * c1 * x1 * x2**2 * c2**2 - 2 * c2 * x1**2 * x2 * c1**2
        numerator -= 4 * c1**2 * x1 * c2 * x2**2 + 2 * c1**2 * x1 * c2 * x2
        numerator += 4 * c2**2 * x2 * c1 * x1**2 - 2 * c2**2 * x2 * c1 * x1
        numerator -= x1**3 * x2 * c1**3 + x1**2 * x2**2 * c2**3
        numerator += x1 * x2**3 * c2**3 - 2 * x1**2 * x2**2 * c2**2 * c1
        numerator += x1**3 * x2 * c2 * c1**2 - x1 * x2**3 * c1 * c2**2
        numerator -= x1**3 * x2 * c2**2 * c1 - x1**2 * x2**2 * c1**3
        numerator += 2 * x1**2 * x2**2 * c1**2 * c2 + x1 * x2**3 * c1**2 * c2

        denominator = c1**4 * x1**3 + 2 * c1**2 * x1**2 * c2**2 * x2
        denominator -= 2 * c1**2 * x1 * c2**2 * x2 + 2 * c2**2 * x2**2 * c1**2 * x1
        denominator += c2**4 * x2**3 - c1**3 * x1 * c2 * x2**2 + c1**3 * x1 * c2 * x2
        denominator -= c2**3 * x2 * c1 * x1**2 + c2**3 * x2 * c1 * x1

        u3 = numerator / denominator

        x0dot = x[0] - x[0] * x[1] - c1 * x[0] * u1
        x1dot = -x[1] + x[0] * x[1] - c2 * x[1] * u1
        lam00dot = 2 * (x[0] - 1) - lam0[0] * (1 - x[1] - c1 * u1) - lam0[1] * x[1]
        lam01dot = 2 * (x[1] - 1) + lam0[0] * x[0] - lam0[1] * (-1 + x[0] - c2 * u1)

        y0dot = y[0] - y[0] * y[1] - c1 * y[0] * u2
        y1dot = -y[1] + y[0] * y[1] - c2 * y[1] * u2
        lam10dot = 2 * (y[0] - 1) - lam1[0] * (1 - y[1] - c1 * u2) - lam1[1] * y[1]
        lam11dot = 2 * (y[1] - 1) + lam1[0] * y[0] - lam1[1] * (-1 + y[0] - c2 * u2)

        z0dot = z[0] - z[0] * z[1] - c1 * z[0] * u3
        z1dot = -z[1] + z[0] * z[1] - c2 * z[1] * u3
        lam20dot = 2 * (z[0] - 1) - lam2[0] * (1 - z[1] - c1 * u3) - lam2[1] * z[1]
        lam21dot = 2 * (z[1] - 1) + lam2[0] * z[0] - lam2[1] * (-1 + z[0] - c2 * u3)

        ts0, ts1 = 0.6, 0.3
        t_rem = (t_final - ts0 * 4 - ts1 * 4) / 4

        sdot = cs.vertcat(x0dot * ts0, x1dot * ts0,
                          lam00dot * ts0, lam01dot * ts0,
                          y0dot * ts1, y1dot * ts1,
                          lam10dot * ts1, lam11dot * ts1,
                          z0dot * t_rem, z1dot * t_rem,
                          lam20dot * t_rem, lam21dot * t_rem,
                          )
        s = cs.vertcat(x, lam0, y, lam1, z, lam2)

        # Objective term
        ode = {'x': s, 'ode': sdot, 't': t}
        return ode

    def get_boundary_fct(self):
        s_start = cs.MX.sym('si', self.x_dim)
        s_end = cs.MX.sym('se', self.x_dim)

        x_start = s_start[:2]
        x_end = s_end[:2]
        lam0_end = s_end[2:4]
        lam0_end = s_end[2:4]

        y_start = s_start[4:6]
        y_end = s_end[4:6]
        lam1_start = s_start[6:8]
        lam1_end = s_end[6:8]

        z_start = s_start[8:10]
        lam2_start = s_start[10:12]
        lam2_end = s_end[10:12]

        # original start point for x
        bvpx = x_start - cs.DM([0.5, 0.7])

        # conditions for interval [t1, t2]
        bvpy = y_start - x_end
        bvplam1 = lam1_start - lam0_end

        # conditions on final interval [t2, t_end]
        bvpz = z_start - y_end
        bvplam2 = lam2_start - lam1_end
        bvplamf = lam2_end

        bvs = cs.vertcat(bvpx, bvpy, bvplam1, bvpz, bvplam2, bvplamf)
        R = cs.Function('R', [s_start, s_end], [bvs])
        return R

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([0.5, 0.7,
                                 5., 1.5,
                                 1., 0.6,
                                 -1, 2,
                                 1., 0.8,
                                 -0., 0.,
                                 ])
        init["s_dim"] = self.x_dim
        return init

    def get_grid_details(self):
        min_t = 0
        max_t = 4
        num_lifts = 20
        return min_t, max_t, num_lifts
