import numpy as np
from . import initialization

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from concurrent.futures import ProcessPoolExecutor

# globals for workers
_global_ode = None


def get_num_items(arr, item):
    """Return how often a certain item appears in a given array.

    Keyword arguments:
        arr -- list of items
        item -- element for which to compute the number of appearances
    """
    counter = 0
    for el in arr:
        if (np.isclose(el, item)):
            counter += 1
    return counter


def remove_nan(arr):
    """Replace all appearances of not a number in an array by infinity.

    Keyword arguments:
        arr -- list of items
    """
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if (np.isnan(arr[i, j])):
                arr[i, j] = np.inf
    return arr


def _compute_state(args):
    # worker function for parallelization
    ode = _global_ode
    i, starting_time, starting_vals, x_dim, time_points = args
    temp_grid = {"time": [el for el in time_points if el >= starting_time]}
    temp_init = {"s_start": starting_vals[i * x_dim:(i + 1) * x_dim], "s_dim": x_dim}
    int_vals = initialization.initialize(temp_init, temp_grid, ode, "auto")
    int_vals = np.reshape(int_vals, (-1, x_dim))
    int_vals = remove_nan(int_vals)
    return i, int_vals


def best_graph_lift(ode, R, starting_times, starting_vals, time_points, x_dim,
                    verbose=False, return_state_indx=False, parallel=True, max_workers=None):
    """Compute the lifting with the optimal residual contraction for a BVP.

    Keyword arguments:
        ode     -- casadi function that describes the ODE
        R       -- boundary function
        starting_times  --  times at which the candidates for the intermediate variables are
                            introduced (ordered list with ascending entries)
        starting_vals   --  candidate values for the intermediate variables that are introduced
                            at the time points given in starting_times
        time_points     --  times for possible lifting points
        x_dim           --  dimension of the states
        verbose         --  print additional information
        return_state_indx   --  return the index of the best new candidate for a lifting point
                                (if there are multiple candidates at one time point)
        parallel        -- enables parallel computation for FSInit
        max_workers     -- maximum number of parallel workers (only if parallel=True)
    """
    states = []
    eps = 1.e-16  # add eps to every edge weight, since edges with weight 0 do not exit

    # integrate all states to the end of the time interval
    if parallel:
        global _global_ode
        _global_ode = ode
        tasks = [(i, starting_times[i], starting_vals, x_dim, time_points)
                 for i in range(len(starting_times))]

        states = [None] * len(starting_times)  # preallocate

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i, int_vals in executor.map(_compute_state, tasks):
                states[i] = int_vals

    else:
        for i in range(len(starting_times)):
            temp_grid = {"time": [el for el in time_points if el >= starting_times[i]]}
            temp_init = {"s_start": starting_vals[i * x_dim:(i + 1) * x_dim], "s_dim": x_dim}
            int_vals = initialization.initialize(temp_init, temp_grid, ode, "auto")
            int_vals = np.reshape(int_vals, (-1, x_dim))
            int_vals = remove_nan(int_vals)
            states += [int_vals]

    num_nodes = 1
    incidence = np.zeros((num_nodes, num_nodes))

    # a "layer" stands for all vertices in the graph that belong to a certain time point
    # keep track of current and next layer
    times = [0]     # keep track of at which time point a candidate was introduced
    candidate_times = [0]
    curr_candidates = 0                  # number of candidates up to current lifting point
    num_lift_points = len(time_points) - 1
    curr_time = 0                       # time corresponding to lifting point
    num_curr_candidates = 0
    if (return_state_indx):
        state_indices = [0]  # keep track of the candidates that were introduced at a time point
        candidate_indices = [0]

    # iterate over all time points / layers
    for curr_lift_point in range(num_lift_points):
        curr_time = time_points[curr_lift_point]
        # compute number of candidates in current layer
        num_curr_candidates += get_num_items(starting_times, curr_time)

        next_time = time_points[curr_lift_point + 1]
        # number of new candidates at the next time point
        num_new_nodes = get_num_items(starting_times, next_time)
        # total number of candidates in next layer
        num_candidates_next_layer = num_curr_candidates + num_new_nodes

        candidate_times += [next_time] * num_new_nodes
        times += candidate_times
        if (return_state_indx):
            candidate_indices += [i for i in range(num_new_nodes)]
            state_indices += candidate_indices

        incidence = np.pad(incidence, ((0, num_candidates_next_layer),
                                       (0, num_candidates_next_layer)),
                           'constant', constant_values=((0, 0), (0, 0)))

        # iterate over all candidates at current lifting point
        for j in range(curr_candidates, curr_candidates + num_curr_candidates):
            # add edge corresponding to no lifting
            incidence[j, num_curr_candidates + j] = eps
            # add distance to all newly introduced nodes
            start_index = curr_candidates + 2 * num_curr_candidates
            stop_index = curr_candidates + 2 * num_curr_candidates + num_new_nodes
            for k in range(start_index, stop_index):
                # total index of the new candidate
                indx = k - curr_candidates - num_curr_candidates
                # value of the old candidate
                s_old = states[j - curr_candidates][-(num_lift_points - curr_lift_point)]
                # value of the new candidate
                s_new = states[indx][0]
                try:
                    diff = s_old - s_new
                    dist = np.linalg.norm(diff)**2
                except RuntimeWarning:
                    dist = np.inf

                if (np.isnan(dist)):
                    dist = np.inf
                incidence[j, k] = dist + eps
        curr_candidates += num_curr_candidates

    num_candidates_last_layer = len(starting_times)
    incidence = np.pad(incidence, ((0, 1), (0, 1)), 'constant', constant_values=((0, 0), (0, 0)))
    # unique independant variable
    start_val = states[0][0]
    # iterate over all candidates at the final lifting point
    start_index = len(incidence) - 1 - num_candidates_last_layer
    for m in range(num_candidates_last_layer):
        end_val = states[m][-1]
        err = np.linalg.norm(R(start_val, end_val))**2
        if (np.isnan(err)):
            err = np.inf
        # add final error of boundary function
        incidence[start_index + m][-1] = err + eps

    graph = csr_matrix(incidence)
    best_lifting_points = get_shortest_path(graph, times, verbose)
    if (return_state_indx):
        best_state_indices = [0]
        for i in range(1, len(best_lifting_points)):
            if (best_lifting_points[i] != best_lifting_points[i - 1]):
                best_state_indices += [state_indices[i]]
        return np.unique(best_lifting_points), best_state_indices

    return np.unique(best_lifting_points)


def get_shortest_path(graph, times, verbose=False):
    """Compute the shortest path in the graph and return the corresponding lifting times.

    Keyword arguments:
        graph   -- graph for which we want to compute a shortest path
        times   -- time points that correspond to the candidates for intermediate variables
    """
    total_times = times + [1]
    dist_matrix, predecessors = shortest_path(csgraph=graph, directed=True, indices=[0],
                                              return_predecessors=True)
    predecessors = list(predecessors.flatten())
    curr_ind = predecessors[-1]
    lifting_points = []
    while (curr_ind >= 0):
        lifting_points = [total_times[curr_ind]] + lifting_points
        curr_ind = predecessors[curr_ind]
    return lifting_points

