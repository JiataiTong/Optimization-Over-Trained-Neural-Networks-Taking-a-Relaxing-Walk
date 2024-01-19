from walk import *
from io_csv import *
from dynamic_new_point import *


def relaxation_test(input_size, layer_num, layer_size, random_seed, walk_eps, pick_bias, timelimit):
    seed = random_seed
    eps = walk_eps
    bias = pick_bias

    layer_dims = layer_num * [layer_size] + [1]
    model_nn = Network(in_size=input_size, layer_dims=layer_dims, seed=seed)
    start = time.time()
    max_ = -1000
    x_max = None
    update_list = []

    np.random.seed(seed)

    time_list = []

    x, frac_z = get_linear_relaxation(model_nn, timelimit)
    if x is None:
        return None, None, None, None, None, None, None
    before = time.time()
    int_z = get_binary_activations(model_nn, x)
    cost = time.time() - before
    time_list.append(cost)
    # print("time_consuming:", cost)
    prob_list = []
    for i in range(layer_num):
        prob_list_layer = get_prob_list_with_bias(int_z[i], frac_z[i], bias)
        prob_list.append(prob_list_layer)

    valid_start_count = 1
    start_count = 1

    # Generate a new points in a while loop
    while time.time() - start < timelimit:

        restriction = []
        picked_neurons = []

        for i in range(layer_num):
            while len(restriction) < len(int_z[i]) and len(picked_neurons) < len(
                    int_z[i]) and time.time() - start < timelimit:
                random_num = np.random.uniform(0, prob_list[i][-1])
                pick = get_index_from_prob_list(prob_list[i], random_num)
                while pick in picked_neurons:
                    random_num = np.random.uniform(0, prob_list[i][-1])
                    pick = get_index_from_prob_list(prob_list[i], random_num)
                picked_neurons.append(pick)
                restriction.append([i, pick])
                # print(restriction)

                # Generate a single new point
                before = time.time()
                x_vals, frac_ap = get_linear_relaxation_with_restriction(model_nn, int_z, restriction)
                cost = time.time() - before
                time_list.append(cost)
                # print("time_consuming:", cost)
                if x_vals is None:
                    # In this case, infeasible, then try to pick another point
                    restriction.remove(restriction[-1])
                    continue
                ap = get_binary_activations(model_nn, x_vals)
                start_count += 1
                time_remain = timelimit - (time.time() - start)

    time_count = time.time() - start
    # store_data([['RW', [input_size] + layer_dims, [walk_eps, pick_bias], random_seed, x_max, max_, first_max, time_count,
    #              start_count, valid_start_count, update_list]], f'RW_experiment_result_{timelimit}.csv')

    return time_list
