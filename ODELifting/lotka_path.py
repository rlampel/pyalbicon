import casadi as cs
import matplotlib.pyplot as plt
import matplotlib as mpl
import utils.create_bvp as create_bvp
import utils.initialization as initialization
import utils.newton_path as newton_path
import apps.lotka_volterra as lv
mpl.rcParams['lines.linewidth'] = 2
mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)


problem = lv.Problem()
ode = problem.get_ode()
R = problem.get_boundary_fct()
min_t, max_t, num_lifts = problem.get_grid_details()

init = problem.get_init()
s_dim = init["s_dim"]

# settings
start = cs.DM([1.225, 0.75])
lamb = 1
custom_init = False
if custom_init:
    custom_init_val = cs.DM([3, 3])
    # custom_init_val = cs.DM([0.2, 0.2])

# define unlifted grid
grid = {}
time_points = [min_t + (max_t - min_t) * i / num_lifts for i in range(num_lifts + 1)]
grid["time"] = time_points
lifting_points = [0 for i in range(len(time_points))]
grid["lift"] = lifting_points

# unlifted version
B_def = create_bvp.create_bvp(ode, R, grid, s_dim)
x_arr_def = newton_path.newton_path(B_def, start, lamb=lamb)

# lifted version
all_states = initialization.initialize_auto(init, grid, ode)
lifting_points[6] = 1
grid["lift"] = lifting_points
B_lift = create_bvp.create_bvp(ode, R, grid, s_dim)

# alternative custom initialization:
if (custom_init):
    lift_start = cs.vertcat(start, custom_init_val)
else:
    lift_start = initialization.select_states(all_states, 2, lifting_points)

x_arr_lift = newton_path.newton_path(B_lift, lift_start, lamb=lamb)

plt.figure(figsize=(8, 6))
plt.plot([float(el[0]) for el in x_arr_def], [float(el[1]) for el in x_arr_def],
         linestyle=":", color="red", label="default", linewidth=2)
plt.plot([float(el[0]) for el in x_arr_lift], [float(el[1]) for el in x_arr_lift],
         linestyle="-.", color="orange", label="lifted init " + str(lift_start[2:]), linewidth=2)

plt.xlabel(r"$x_0$", fontsize=18)
plt.ylabel(r"$x_1$", fontsize=18)
plt.title("Discrete Newton path with " + r"$\lambda = " + str(lamb) + "$", fontsize=24)
plt.xlim(0, 1.5)
plt.ylim(0, 1.5)
plt.legend(fontsize=14)
plt.scatter(float(x_arr_def[0][0]), float(x_arr_def[0][1]), marker="x", color="black")
plt.scatter(float(x_arr_def[-1][0]), float(x_arr_def[-1][1]), marker="x", color="black")
plt.text(float(x_arr_def[0][0]) + 0.025, float(x_arr_def[0][1]) + 0.025, "start", fontsize=14)
plt.text(float(x_arr_def[-1][0]) + 0.025, float(x_arr_def[-1][1]) - 0.05, "root", fontsize=14)
plt.show()
