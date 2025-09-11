import matplotlib.pyplot as plt
import casadi as cs
import utils.newton as newton
import utils.create_poly as create_poly
import utils.initialize as initialize
import os


poly_dim = 17
start = cs.DM([5., 0.])
coeffs = [-2] + [0] * 15 + [1]
num_reps = 1

# settings
log_results = True  # chose whether to write the iterations to a file
log_type = "res"  # choose "res" to plot the residual convergence
# log_type = "step"  # otherwise the convergence w.r.t. step lenghts is plotted
newton_opts = {"verbose": False, "max_iter": 100, "log_type": log_type}

if log_results:
    dirname = os.path.dirname(__file__)

if (log_type == "step"):
    name_suffix = "_step"
else:
    name_suffix = ""

if (log_results):
    # clear logged results
    curr_name = "poly_data/x16_default" + name_suffix + ".dat"
    curr_path = os.path.join(dirname, curr_name)
    f = open(curr_path, "w")
    f.write("")
    f.close()
    for exponent in range(6):
        curr_name = "poly_data/x16_lift_sqrt[" + str(2**exponent) + "]2" + name_suffix + ".dat"
        curr_path = os.path.join(dirname, curr_name)
        f = open(curr_path, "w")
        f.write("")
        f.close()

# line styles for plotting
line_styles = ["-.", "--", ":", "-"]

for i in range(num_reps):

    F = create_poly.create_poly(coeffs)
    lift_sol, lift_conv = newton.newton(F, start, newton_opts)
    if (log_type == "step"):
        lift_conv = [float(cs.norm_2(el[:2])) for el in lift_conv]
    plt.plot([i for i in range(len(lift_conv))], lift_conv, label="default")

    print("first contraction: " + str(lift_conv[1] / lift_conv[0]) + "\n")
    if (log_results):
        curr_name = "poly_data/x16_default" + name_suffix + ".dat"
        curr_path = os.path.join(dirname, curr_name)
        f = open(curr_path, "a")
        f.write("no lifting: \n")
        for i in range(0, len(lift_conv)):
            el = str(lift_conv[i])
            f.write(str(i) + " " + el + "\n")
        f.write("\n")
        f.close()

    for exponent in range(0, 6):
        lift_degree = 2 ** (1 / 2**exponent)
        print("component degree is ", lift_degree)
        # create lifting with fixed polynomial degree
        lift_start = initialize.initialize_auto(start, poly_dim, lift_degree)
        lift_start = lift_start[:8 * 2**(exponent)]
        lifting_points = [1] * 4 * 2**(exponent)
        lifting_points += [0]
        G = create_poly.create_lifted_poly(coeffs, lift_degree, lifting_points=lifting_points)
        lift_sol, lift_conv = newton.newton(G, lift_start, newton_opts)
        if (log_type == "step"):
            lift_conv = [float(cs.norm_2(el[:2 * 4 * 2**(exponent)])) for el in lift_conv]

        if exponent == 0:
            curr_label = r"lifted degree $2$"
        else:
            curr_label = r"lifted degree $\sqrt[" + str(2**exponent) + "]{2}$"

        print("first contraction: " + str(lift_conv[1] / lift_conv[0]) + "\n")
        if (log_results):
            curr_name = "poly_data/x16_lift_sqrt[" + str(2**exponent) + "]2" + name_suffix + ".dat"
            curr_path = os.path.join(dirname, curr_name)
            f = open(curr_path, "a")
            f.write("exponent is " + str(exponent) + "\n")
            for i in range(0, len(lift_conv)):
                el = str(lift_conv[i])
                f.write(str(i) + " " + el + "\n")
            f.write("\n")
            f.close()

        plt.plot([i for i in range(len(lift_conv))], lift_conv, label=curr_label,
                 linestyle=line_styles[exponent % 4])
    plt.yscale("log")
    plt.xlabel("iteration")
    if log_type == "res":
        plt.ylabel("residual norm")
    else:
        plt.ylabel("step size")
    plt.legend()
    plt.show()

