from walk import *
from io_csv import *
from dynamic_new_point import *


def relaxation_walk(input_size, layer_num, layer_size, random_seed, walk_eps, pick_bias, timelimit):
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

    x, frac_z = get_linear_relaxation(model_nn, timelimit)
    if x is None:
        store_data([['RW', [input_size] + layer_dims, [walk_eps, pick_bias], random_seed, None, None, None, None, None,
                     None, None]], f'RW_experiment_result_{timelimit}.csv')
        return None, None, None, None, None, None, None
    int_z = get_binary_activations(model_nn, x)
    prob_list_first_layer = get_prob_list_with_bias(int_z[0], frac_z[0], bias)

    prob_list_second_layer = []
    if layer_num >= 2:
        prob_list_second_layer = get_prob_list_with_bias(int_z[1], frac_z[1], bias)

    # First walk for relaxation
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
                local_x_max, local_max, step_count1, time_consuming = single_walk_with_timelimit(model_nn, x_vals, eps,
                                                                                                 time_remain)
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
            # print(f'end with 1 loop at {time.time() - start}')
            sys.stdout.flush()

    time_count = time.time() - start
    # store_data([['RW', [input_size] + layer_dims, [walk_eps, pick_bias], random_seed, x_max, max_, first_max, time_count,
    #              start_count, valid_start_count, update_list]], f'RW_experiment_result_{timelimit}.csv')
    if max_ is None:
        store_data([['RW', [input_size] + layer_dims, [walk_eps, pick_bias], random_seed, x_max,
                     None, first_max, time_count,
                     start_count, valid_start_count, update_list]], f'RW_experiment_result_{timelimit}.csv')
    else:
        store_data([['RW', [input_size] + layer_dims, [walk_eps, pick_bias], random_seed, x_max, model_nn(torch.FloatTensor(x_max)).item(), first_max, time_count,
                     start_count, valid_start_count, update_list]], f'RW_experiment_result_{timelimit}.csv')
    return x_max, max_, first_max, time_count, start_count, valid_start_count, update_list


def sampling_mip(input_size, layer_num, layer_size, random_seed, gap, timelimit):
    seed = random_seed
    seconds = timelimit

    layer_dims = layer_num * [layer_size] + [1]
    model_nn = Network(in_size=input_size, layer_dims=layer_dims, seed=seed)

    np.random.seed(seed)
    start = time.time()
    count = 0
    valid_start_count = 0
    max_ = -1000
    x_max = None
    first_max = -1000
    update_list = []
    step_list = []
    step_val_list = []

    while time.time() - start < seconds:
        isUpdated = False
        count = count + 1
        ap = get_activations(model_nn, np.random.rand(model_nn.in_size))
        max_lp, x_new, pre_activation_output = solve_lp_enhanced(model_nn, ap)
        step_count = 1
        step_vals = []
        step_vals.append(max_lp)
        if max_lp > max_:
            max_ = max_lp
            x_max = x_new
            isUpdated = True
        if np.abs((max_ - max_lp) / max_) < gap:
            valid_start_count += 1
            while 1:
                time_remain = seconds - (time.time() - start)
                # print(f'{count}: time limit: {time_remain}')
                if time_remain <= 0:
                    if isUpdated:
                        now = time.time() - start
                        update_list.append([max_, x_max, count, valid_start_count, now])
                    break
                max_ip, x_new, pre_activation_output = solve_mip(model_nn, ap, pre_activation_output, time_remain)
                step_count += 1

                step_vals.append(max_ip)
                if x_new is None:
                    # In this case, the time for the last solve mip is too short, no solution found.
                    print(f'{count} time out')
                    break
                ap = get_activations(model_nn, x_new)
                # print(f'{count}: mip result: {max_ip}')
                if max_ip > max_lp:
                    max_lp = max_ip
                    if max_lp > max_:
                        max_ = max_lp
                        x_max = x_new
                        isUpdated = True
                else:
                    if isUpdated:
                        now = time.time() - start
                        update_list.append([max_, x_max, count, valid_start_count, now])
                    break
            step_list.append(step_count)
            step_val_list.append(step_vals)
        if count == 1:
            first_max = max_
    now = time.time() - start
    if max_ is None:
        store_data([['SM', [input_size] + layer_dims, [gap], random_seed, x_max,
                     None, first_max, now, count, valid_start_count, update_list]],
                   f'SM_experiment_result_{timelimit}.csv')
    else:
        # store_data([['SM', [input_size] + layer_dims, [gap], random_seed, x_max, max_, first_max, now, count, valid_start_count, update_list]],
        #            f'SM_experiment_result_{timelimit}.csv')
        store_data([['SM', [input_size] + layer_dims, [gap], random_seed, x_max, model_nn(torch.FloatTensor(x_max)).item(), first_max, now, count, valid_start_count, update_list]],
                   f'SM_experiment_result_{timelimit}.csv')
    return x_max, max_, first_max, now, count, valid_start_count, update_list


def relaxation_walk_dynamic(input_size, layer_num, layer_size, random_seed, walk_eps, timelimit):
    seed = random_seed
    eps = walk_eps
    # bias = pick_bias

    layer_dims = layer_num * [layer_size] + [1]
    model_nn = Network(in_size=input_size, layer_dims=layer_dims, seed=seed)
    start = time.time()
    max_ = -1000
    x_max = None
    update_list = []
    record_ap = []

    start_count = 1
    valid_start_count = 1

    np.random.seed(seed)

    x, z = get_linear_relaxation(model_nn, timelimit)
    if x is None:
        store_data([['RWD', [input_size] + layer_dims, [walk_eps], random_seed, None, None, None, None, None,
                     None, None]], f'RWD_experiment_result_{timelimit}.csv')
        return None, None, None, None, None, None, None
    ap = get_binary_activations(model_nn, x)
    record_ap.append(ap)

    np.random.seed(seed)
    z_round = round_list(z)
    dif_index = different_index(ap, z_round)
    pick = np.random.randint(len(dif_index))
    new_x, new_z = find_new_relaxation_point_random(model_nn, x, pick, dif_index)
    while new_x is None:
        pick = np.random.randint(len(dif_index))
        new_x, new_z = find_new_relaxation_point_random(model_nn, x, pick, dif_index)

    time_remain = timelimit - (time.time() - start)
    x_max_old, max_old, step_count_old, time_consuming_old = single_walk_with_timelimit(model_nn, x, eps, time_remain)
    now = time.time() - start
    update_list.append([max_old, x_max_old, 1, 1, now])
    # print(f'relaxation: {max_old}')
    # max_list = [max_old]
    max_ = max_old
    while time.time() - start < timelimit:
        new_ap = get_binary_activations(model_nn, new_x)
        record_ap.append(ap)
        new_z_round = round_list(new_z)
        dif_index = different_index(new_z_round, new_ap)
        pick = np.random.randint(len(dif_index))
        temp_x = new_x
        temp_z = new_z
        new_x, new_z = find_new_relaxation_point_random(model_nn, new_x, pick, dif_index)
        start_count += 1
        if new_x is None:
            # print('INFEASIBLE: Can not find more points')
            # Find another point
            new_x = temp_x
            new_z = temp_z
            continue
        if new_ap in record_ap:
            continue
        time_remain = timelimit - (time.time() - start)
        x_max_new, max_new, step_count_new, time_consuming_new = single_walk_with_timelimit(model_nn, temp_x, eps, time_remain)
        valid_start_count += 1
        # print(f'{max_new}')
        # max_list.append(max_new)
        if max_new > max_:
            x_max = x_max_new
            max_ = max_new
            now = time.time() - start
            update_list.append([max_, x_max, start_count, valid_start_count, now])
    # print('max: ', max(max_list))
    now = time.time() - start
    # store_data([['RWD', [input_size] + layer_dims, [walk_eps], random_seed, x_max, max_, max_old, now, start_count,
    #              valid_start_count, update_list]],
    #            f'RWD_experiment_result_{timelimit}.csv')
    store_data([['RWD', [input_size] + layer_dims, [walk_eps], random_seed, x_max, model_nn(torch.FloatTensor(x_max)).item(), max_old, now, start_count,
                 valid_start_count, update_list]],
               f'RWD_experiment_result_{timelimit}.csv')
    return x_max, max_, max_old, now, start_count, valid_start_count, update_list


if __name__ == '__main__':
    if len(sys.argv) < 8:
        print(f'please give a valid function')
    elif sys.argv[1] == 'relaxation_walk':
        assert len(sys.argv) == 9
        relaxation_walk(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]), int(sys.argv[8]))
    elif sys.argv[1] == 'sampling_mip':
        assert len(sys.argv) == 8
        sampling_mip(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7]))
    elif sys.argv[1] == 'relaxation_walk_dynamic':
        assert len(sys.argv) == 8
        relaxation_walk_dynamic(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7]))
    else:
        print(f'please give a valid function')
