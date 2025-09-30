import casadi as cs
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os


# Define the neural network

lift = True    # automatically lifted (True) or unlifted (False) formulation
log_results = True  # write the residual convergence to a file
dirname = os.path.dirname(__file__)
filename_log = os.path.join(dirname, "nn_results" + "_lift" * lift + ".log")
target_index = 9

# set initial perturbartion, i.e., start point
start = cs.DM([0.1] * 784)

if (log_results):
    f = open(filename_log, "w")
    title = "Lifted" if lift else "Unlifted"
    f.write(title + "\n")
    f.close()


# ----------------------------------------------------------------------------------------
# function definitions

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 64)
        self.layer4 = nn.Linear(64, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = self.sigmoid(self.layer1(x))
        x = self.layer4(x)  # No activation on the output layer
        return x


def sigmoid(inputs):
    return 1 / (1 + cs.exp(-inputs))


def sigmoid_der(inputs):
    X = cs.MX.sym("X", inputs.shape[0])
    Y = sigmoid(X)
    S = cs.Function("S", [X], [Y], ["X"], ["Y"])
    DS = cs.Function("DS", [X], [cs.jacobian(S(X), X)], ["X"], ["DS"])
    return DS(inputs)


def cross_entropy_loss(output, target_index):
    out_dim = output.shape[0]
    exp_sum = 0
    for i in range(out_dim):
        exp_sum += cs.exp(output[i])
    nom = cs.exp(output[target_index])
    return -cs.log(nom / exp_sum)


def get_weights(model):
    W = []
    b = []
    counter = 0
    for param in model.parameters():
        curr_weight = cs.DM(param.data.numpy())
        if (counter % 2 == 0):
            # add weight matrix
            W += [curr_weight]
        else:
            # add bias
            b += [curr_weight]
        counter += 1
    return W, b


def compute_final_output(model, input):
    Xk = input
    Xk += seven_image

    W, b = get_weights(model)
    Xk = sigmoid(W[0] @ Xk + b[0])
    Xk = W[1] @ Xk + b[1]
    return Xk


def create_lifted_init(model, input):
    lifted_init = input
    Xk = input
    Xk += seven_image

    W, b = get_weights(model)
    Xk = sigmoid(W[0] @ Xk + b[0])
    lifted_init = cs.vertcat(lifted_init, Xk)
    return lifted_init


def create_adversary(model, target_index, lift=True):
    X = cs.MX.sym("X", 784)
    total_input = [X]
    RHS = cs.DM([])
    Xk = X

    # add starting image
    Xk += seven_image
    W, b = get_weights(model)

    # lift after the sigmoid function
    Xk_temp = W[0] @ Xk + b[0]
    Xk_old = sigmoid(Xk_temp)
    if (lift):
        curr_size = Xk_old.shape[0]
        Xk = cs.MX.sym("X1", curr_size)
        total_input += [Xk]
        RHS = cs.vertcat(RHS, Xk_old - Xk)
    else:
        Xk = Xk_old
    Xk_new = W[1] @ Xk + b[1]

    # loss function
    temp_var = cs.MX.sym("temp", 10)
    ce_loss = cross_entropy_loss(temp_var, target_index)
    L = cs.Function("L", [temp_var], [ce_loss])
    DL = cs.Function("DL", [temp_var], [cs.jacobian(L(temp_var), temp_var)])

    # penalty function
    one_diff = X - cs.DM([1] * 784)
    P1 = cs.norm_2(one_diff)**2
    P2 = cs.norm_2(X)**2
    P = cs.Function("P", [X], [P1 + P2])
    DP = cs.Function("DP", [X], [cs.jacobian(P(X), X)])

    # output of final layer
    final_layer = DL(Xk_new) @ W[1]
    final_layer @= sigmoid_der(Xk_temp) @ W[0]

    total_input = cs.vertcat(*total_input)
    final_layer += DP(X)
    RHS = cs.vertcat(RHS, final_layer.T)
    return cs.Function("NN", [total_input], [RHS])


def newton(G, x_start, model, opts={}):
    TOL = opts.get("TOL", 1.e-10)
    max_iter = opts.get("max_iter", 100)

    x = x_start
    counter = 0

    func_norm = cs.norm_2(G(x))
    print("starting with norm: ", func_norm)
    func_arr = [func_norm]

    nn_rootfinder = cs.rootfinder('nn_root', 'newton', DNet,
                                  {'line_search': False, 'max_iter': 1, 'error_on_fail': False})
    while (func_norm > TOL and counter < max_iter):
        x = nn_rootfinder(x)

        func_val = G(x)
        func_norm = cs.norm_2(func_val)
        func_arr += [func_norm]
        counter += 1

        print("Iteration: ", counter)
        print("\t norm: ", func_norm)
        curr_out = compute_final_output(model, x[:784])
        plot_sol = np.reshape(np.array(x[:784]), (28, 28))
        plot_sol += plot_seven
        plt.clf()
        plt.title("Iteration " + str(counter))
        plt.imshow(plot_sol, cmap="Greys")
        plt.colorbar()
        plt.pause(0.2)
        if (log_results):
            f = open(filename_log, "a")
            f.write(str(counter) + " " + str(func_norm) + "\n")
            f.close()
        print("\t Current loss: ", cross_entropy_loss(curr_out, target_index))
        # print("Sigmoid: ", sigmoid(curr_out))
    return x, func_arr


def auto_lifted_newton(G, x_start, model, opts={}):
    TOL = opts.get("TOL", 1.e-10)
    max_iter = opts.get("max_iter", 100)

    x = x_start
    counter = 0

    func_norm = cs.norm_2(G(x))
    print("starting with norm: ", func_norm)
    func_arr = [func_norm]

    nn_rootfinder = cs.rootfinder('nn_root', 'newton', DNet,
                                  {'line_search': False, 'max_iter': 1, 'error_on_fail': False})
    while (func_norm > TOL and counter < max_iter):
        x = nn_rootfinder(x)

        # test whether FSInit is better
        func_val = G(x)
        func_norm = cs.norm_2(func_val)

        x_fs = create_lifted_init(model, x[:784])
        fs_val = G(x_fs)
        fs_norm = cs.norm_2(fs_val)

        if func_norm > fs_norm:
            # REPLACE BY FSINIT
            x = x_fs
            func_norm = fs_norm

        func_norm = cs.norm_2(func_val)
        func_arr += [func_norm]
        counter += 1

        print("Iteration: ", counter)
        print("\t norm: ", func_norm)
        curr_out = compute_final_output(model, x[:784])
        plot_sol = np.reshape(np.array(x[:784]), (28, 28))
        plot_sol += plot_seven
        plt.clf()
        plt.title("Iteration " + str(counter))
        plt.imshow(plot_sol, cmap="Greys")
        plt.colorbar()
        plt.pause(0.2)
        if (log_results):
            f = open(filename_log, "a")
            f.write(str(counter) + " " + str(func_norm) + "\n")
            f.close()
        print("\t Current loss: ", cross_entropy_loss(curr_out, target_index))
    return x, func_arr


# ----------------------------------------------------------------------------------------
# load the model weights
model = NeuralNet()
filename_weights = os.path.join(dirname, "trained_weights/mnist_model.pth")
model.load_state_dict(torch.load(filename_weights))

filename_seven = os.path.join(dirname, "MNIST_7.dat")
seven_image = np.loadtxt(filename_seven, delimiter=",")
seven_image = cs.DM(seven_image)
plot_seven = np.reshape(np.array(seven_image), (28, 28))

# check initial cross entropy loss
out = compute_final_output(model, start)
net_out = out[-10:]
print("Initial output: \t", net_out)
print("Initial cross entropy loss for 9: ", cross_entropy_loss(net_out, 9))


# ----------------------------------------------------------------------------------------
# create and solve the problem

if (lift is True):
    # Create lifted problem
    print("Using lifted function")
    DNet = create_adversary(model, target_index, True)
    start = create_lifted_init(model, start)
    sol, _ = auto_lifted_newton(DNet, start, model)

else:
    print("Using unlifted function")
    # Create unlifted problem
    DNet = create_adversary(model, target_index, False)

    sol, _ = newton(DNet, start, model)

# ----------------------------------------------------------------------------------------
# show final results
plot_sol = np.reshape(np.array(sol[:784]), (28, 28))
plt.clf()
plt.imshow(plot_sol + plot_seven, cmap="Greys")
plt.colorbar()
plt.title("Final solution")
plt.show()
# evaluate final result:
sol = sol[:784]
out = compute_final_output(model, sol)
net_out = out[-10:]
print("Final output: \t", net_out)
print("Final cross entropy loss for 9: ", cross_entropy_loss(net_out, 9))

