from lp import *
from mnist_io_csv import *
import torch
import time
from mnist_util import *
from relaxation import *
from lp import *
from matplotlib import pyplot as plt


def single_walk_with_timelimit(model_nn, x, image, wrong_label, right_label, delta, timelimit):
    # print(x)
    start = time.time()
    max_ = -1000
    x_max = None
    step_count = 0
    ap = get_binary_activations(model_nn, x)
    max_lp, x_new, g = solve_lp_with_g(model_nn, image, wrong_label, right_label, ap, delta)
    while max_lp > max_:
        # print(max_lp)
        step_count += 1
        max_ = max_lp
        x_max = x_new
        now = time.time()
        if now - start >= timelimit:
            break
        ap = update_ap_random(g, ap)
        max_lp, x_new, g = solve_lp_with_g(model_nn, image, wrong_label, right_label, ap, delta)
    time_consuming = time.time() - start
    return x_max, max_, step_count, time_consuming


def relaxation_walk_nmist(model_nn, image, image_index, wrong_label, right_label, delta, random_seed, pick_bias, timelimit, info):
    seed = random_seed
    bias = pick_bias

    input_size, layer_dims = get_model_info(model_nn)
    layer_num = len(layer_dims) - 1
    start = time.time()
    max_ = -1000
    x_max = None
    update_list = []

    np.random.seed(seed)

    x, frac_z = get_linear_relaxation(model_nn, image, wrong_label, right_label, delta, timelimit)
    # print(x)
    # xlist = np.array(list(x.values()))
    # plt.imshow(xlist.reshape((28, 28)), cmap="gray")
    # plt.show()

    if x is None:
        store_data([['RW', [input_size] + layer_dims, [pick_bias], random_seed, image_index, None, None, None, None, None,
                     None, None, None, right_label, wrong_label, info]], f'RW_MNIST_result_{timelimit}_{info[-1]}.csv')
        return None, None, None, None, None, None, None
    int_z = get_binary_activations(model_nn, x)
    # print(int_z)
    prob_list = []
    for i in range(layer_num):
        prob_list_layer = get_prob_list_with_bias(int_z[i], frac_z[i], bias)
        prob_list.append(prob_list_layer)


    # First walk for relaxation
    step_count = 0
    ap = get_binary_activations(model_nn, x)
    # print(ap)
    max_lp, x_new, g_vals = solve_lp_with_g(model_nn, image, wrong_label, right_label, ap, delta)
    while max_lp > max_:
        # print(max_lp)
        step_count += 1
        max_ = max_lp
        x_max = x_new
        if time.time() - start > timelimit:
            break
        ap = update_ap_random(g_vals, ap)
        max_lp, x_new, g_vals = solve_lp_with_g(model_nn, image, wrong_label, right_label, ap, delta)

    first_max = max_
    record_ap = [int_z]
    ap = int_z
    now = time.time() - start
    update_list.append([first_max, x_max, 1, 1, now])

    # print(f'relaxation walk done by {now}')

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
                x_vals, frac_ap = get_linear_relaxation_with_restriction(model_nn, image, wrong_label, right_label, delta, timelimit, ap, restriction)
                if x_vals is None:
                    # In this case, infeasible, then try to pick another point
                    restriction.remove(restriction[-1])
                    continue
                ap = get_binary_activations(model_nn, x_vals)
                start_count += 1
                if ap in record_ap:
                    continue
                valid_start_count += 1
                record_ap.append(ap)
                time_remain = timelimit - (time.time() - start)
                if time_remain <= 0:
                    break
                else:
                    # now = time.time() - start
                    # print(f'{valid_start_count} walk start with time remain{time_remain} at {now}')
                    local_x_max, local_max, step_count1, time_consuming = single_walk_with_timelimit(
                        model_nn, x, image, wrong_label, right_label, delta, timelimit)
                    # now = time.time() - start
                    # print(f'{valid_start_count} walk done by {now} with {step_count1} steps')
                    if local_max > max_:
                        max_ = local_max
                        x_max = local_x_max
                        now = time.time() - start
                        update_list.append([max_, x_max, start_count, valid_start_count, now])


    time_count = time.time() - start
    # store_data([['RW', [input_size] + layer_dims, [walk_eps, pick_bias], random_seed, x_max, max_, first_max, time_count,
    #              start_count, valid_start_count, update_list]], f'RW_experiment_result_{timelimit}.csv')
    if max_ is None:
        store_data([['RW', [input_size] + layer_dims, [pick_bias], random_seed, image_index, x_max,
                     None, first_max, time_count,
                     start_count, valid_start_count, update_list, None, right_label, wrong_label, info]], f'RW_MNIST_result_{timelimit}_{info[-1]}.csv')
    else:
        res = model_nn(torch.FloatTensor(x_max)).tolist()
        store_data([['RW', [input_size] + layer_dims, [pick_bias], random_seed, image_index, x_max, res[wrong_label] - res[right_label], first_max, time_count,
                     start_count, valid_start_count, update_list, res, right_label, wrong_label, info]], f'RW_MNIST_result_{timelimit}_{info[-1]}.csv')
    return x_max, max_, first_max, time_count, start_count, valid_start_count, update_list

