from walk import *
from io_csv import *
from dynamic_new_point import *


def sampling_walk(input_size, layer_num, layer_size, random_seed, walk_eps, pick_bias, timelimit):
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

    x = np.random.rand(model_nn.in_size)

    int_z = get_binary_activations(model_nn, x)

    # First walk for sampling
    step_count = 0
    ap = get_binary_activations(model_nn, x)
    max_lp, x_new = solve_lp_pre_calc(model_nn, ap)
    while max_lp > max_:
        # print(max_lp)
        step_count += 1
        max_ = max_lp
        x_max = x_new
        if time.time() - start > timelimit:
            break
        x = update_x(model_nn, x, x_new, eps)
        ap = get_binary_activations(model_nn, x)
        max_lp, x_new = solve_lp_pre_calc(model_nn, ap)

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

        # Generate a single new point
        x_vals = np.random.rand(model_nn.in_size)
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
            local_x_max, local_max, step_count1, time_consuming = single_walk_with_timelimit(model_nn, x_vals,
                                                                                             eps,
                                                                                             time_remain)
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
        store_data([['SW', [input_size] + layer_dims, [walk_eps, pick_bias], random_seed, x_max,
                     None, first_max, time_count,
                     start_count, valid_start_count, update_list]], f'SW_experiment_result_{timelimit}.csv')
    else:
        store_data([['SW', [input_size] + layer_dims, [walk_eps, pick_bias], random_seed, x_max,
                     model_nn(torch.FloatTensor(x_max)).item(), first_max, time_count,
                     start_count, valid_start_count, update_list]], f'SW_experiment_result_{timelimit}.csv')
    return x_max, max_, first_max, time_count, start_count, valid_start_count, update_list


def relaxation_random_walk(input_size, layer_num, layer_size, random_seed, pick_bias, timelimit):
    seed = random_seed
    bias = pick_bias

    layer_dims = layer_num * [layer_size] + [1]
    model_nn = Network(in_size=input_size, layer_dims=layer_dims, seed=seed)
    start = time.time()
    max_ = -1000
    x_max = None
    update_list = []

    np.random.seed(seed)

    x, frac_z = get_linear_relaxation(model_nn, timelimit)
    if x is None:
        store_data([['RRW', [input_size] + layer_dims, [pick_bias], random_seed, None, None, None, None, None,
                     None, None]], f'RRW_experiment_result_{timelimit}.csv')
        return None, None, None, None, None, None, None
    int_z = get_binary_activations(model_nn, x)
    prob_list_first_layer = get_prob_list_with_bias(int_z[0], frac_z[0], bias)

    prob_list_second_layer = []
    if layer_num >= 2:
        prob_list_second_layer = get_prob_list_with_bias(int_z[1], frac_z[1], bias)

    # First walk for relaxation
    step_count = 0
    ap = get_binary_activations(model_nn, x)
    max_lp, x_new, g = solve_lp_pre_calc_with_g(model_nn, ap)
    while max_lp > max_:
        # print(max_lp)
        step_count += 1
        max_ = max_lp
        x_max = x_new
        if time.time() - start > timelimit:
            break
        ap = update_ap_random(g, ap)
        max_lp, x_new, g = solve_lp_pre_calc_with_g(model_nn, ap)

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
        while len(restriction) < len(int_z[0]) and len(picked_neurons) < len(
                int_z[0]) and time.time() - start < timelimit:
            random_num = np.random.uniform(0, prob_list_first_layer[-1])
            pick = get_index_from_prob_list(prob_list_first_layer, random_num)
            while pick in picked_neurons:
                random_num = np.random.uniform(0, prob_list_first_layer[-1])
                pick = get_index_from_prob_list(prob_list_first_layer, random_num)
            picked_neurons.append(pick)
            restriction.append([0, pick])
            # print(restriction)

            # Generate a single new point
            x_vals, frac_ap = get_linear_relaxation_with_restriction(model_nn, int_z, restriction)
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
                local_x_max, local_max, step_count1, time_consuming = single_random_walk_with_timelimit(model_nn, x_vals, time_remain)
                # now = time.time() - start
                # print(f'{valid_start_count} walk done by {now} with {step_count1} steps')
                if local_max > max_:
                    max_ = local_max
                    x_max = local_x_max
                    now = time.time() - start
                    update_list.append([max_, x_max, start_count, valid_start_count, now])
        if layer_num >= 2:
            # print(f'Start to add restrictions on layer 2 at {time.time() - start}')
            sys.stdout.flush()
            picked_neurons = []
            number_of_restriction_l1 = len(restriction)
            assert prob_list_second_layer != []
            while len(restriction) - number_of_restriction_l1 < len(int_z[1]) and len(picked_neurons) < len(int_z[1]) \
                    and time.time() - start < timelimit:
                # print(f'picked neuron #: {len(picked_neurons)}')
                random_num = np.random.uniform(0, prob_list_second_layer[-1])
                pick = get_index_from_prob_list(prob_list_second_layer, random_num)
                while pick in picked_neurons:
                    random_num = np.random.uniform(0, prob_list_second_layer[-1])
                    pick = get_index_from_prob_list(prob_list_second_layer, random_num)
                picked_neurons.append(pick)
                restriction.append([1, pick])
                # print(restriction)

                # Generate a single new point
                x_vals, frac_ap = get_linear_relaxation_with_restriction(model_nn, int_z, restriction)
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
                    local_x_max, local_max, step_count1, time_consuming = single_random_walk_with_timelimit(model_nn, x_vals, time_remain)
                    # now = time.time() - start
                    # print(f'{valid_start_count} walk done by {now} with {step_count1} steps')
                    if local_max > max_:
                        max_ = local_max
                        x_max = local_x_max
                        now = time.time() - start
                        update_list.append([max_, x_max, start_count, valid_start_count, now])
            # print(f'end with 1 loop at {time.time() - start}')
            sys.stdout.flush()

    time_count = time.time() - start
    # store_data([['RW', [input_size] + layer_dims, [walk_eps, pick_bias], random_seed, x_max, max_, first_max, time_count,
    #              start_count, valid_start_count, update_list]], f'RW_experiment_result_{timelimit}.csv')
    if max_ is None:
        store_data([['RRW', [input_size] + layer_dims, [pick_bias], random_seed, x_max,
                     None, first_max, time_count,
                     start_count, valid_start_count, update_list]], f'RRW_experiment_result_{timelimit}.csv')
    else:
        store_data([['RRW', [input_size] + layer_dims, [pick_bias], random_seed, x_max, model_nn(torch.FloatTensor(x_max)).item(), first_max, time_count,
                     start_count, valid_start_count, update_list]], f'RRW_experiment_result_{timelimit}.csv')
    return x_max, max_, first_max, time_count, start_count, valid_start_count, update_list





def relaxation_random_walk_deep(input_size, layer_num, layer_size, random_seed, pick_bias, timelimit):
    seed = random_seed
    bias = pick_bias

    layer_dims = layer_num * [layer_size] + [1]
    model_nn = Network(in_size=input_size, layer_dims=layer_dims, seed=seed)
    start = time.time()
    max_ = -1000
    x_max = None
    update_list = []

    np.random.seed(seed)

    x, frac_z = get_linear_relaxation(model_nn, timelimit)
    if x is None:
        store_data([['RRW', [input_size] + layer_dims, [pick_bias], random_seed, None, None, None, None, None,
                     None, None]], f'RRW_experiment_result_{timelimit}.csv')
        return None, None, None, None, None, None, None
    int_z = get_binary_activations(model_nn, x)
    prob_list = []
    for i in range(layer_num):
        prob_list_layer = get_prob_list_with_bias(int_z[i], frac_z[i], bias)
        prob_list.append(prob_list_layer)


    # First walk for relaxation
    step_count = 0
    ap = get_binary_activations(model_nn, x)
    max_lp, x_new, g = solve_lp_pre_calc_with_g(model_nn, ap)
    while max_lp > max_:
        # print(max_lp)
        step_count += 1
        max_ = max_lp
        x_max = x_new
        if time.time() - start > timelimit:
            break
        ap = update_ap_random(g, ap)
        max_lp, x_new, g = solve_lp_pre_calc_with_g(model_nn, ap)

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
                x_vals, frac_ap = get_linear_relaxation_with_restriction(model_nn, int_z, restriction)
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
                    local_x_max, local_max, step_count1, time_consuming = local_x_max, local_max, step_count1, time_consuming = single_random_walk_with_timelimit(model_nn, x_vals, time_remain)
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
        store_data([['RRW', [input_size] + layer_dims, [pick_bias], random_seed, x_max,
                     None, first_max, time_count,
                     start_count, valid_start_count, update_list]], f'RRW_experiment_result_{timelimit}.csv')
    else:
        store_data([['RRW', [input_size] + layer_dims, [pick_bias], random_seed, x_max, model_nn(torch.FloatTensor(x_max)).item(), first_max, time_count,
                     start_count, valid_start_count, update_list]], f'RRW_experiment_result_{timelimit}.csv')
    return x_max, max_, first_max, time_count, start_count, valid_start_count, update_list
