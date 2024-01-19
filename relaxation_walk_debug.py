from walk import *
from io_csv import *
from dynamic_new_point import *
def get_linear_relaxation_debug(modelnn, time_limit):
    w = modelnn.get_weight_matrix()
    b = modelnn.get_bias_matrix()
    layer_dims = modelnn.layer_dims
    in_size = modelnn.in_size

    model = gb.Model()
    global x_vals
    global z_vals
    x_vals = []
    z_vals = []

    def get_relaxation(model, where):
        if where == GRB.Callback.MIPNODE:
            global x_vals
            global z_vals
            status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
            if status == GRB.INTEGER:
                x_vals = model.cbGetSolution(x)
                z_vals = []
                for i in range(len(layer_dims)):
                    z_vals.append(model.cbGetNodeRel(z[i]))
                z_vals = model.cbGetSolution(z)
                model.terminate()
            if status == GRB.OPTIMAL:
                x_vals = model.cbGetNodeRel(x)
                z_vals = []
                for i in range(len(layer_dims)):
                    z_vals.append(model.cbGetNodeRel(z[i]))
                model.terminate()

    model.setParam('OutputFlag', 1)
    model.setParam('TimeLimit', time_limit)
    x = model.addVars(in_size, ub=1, lb=0, name="input")
    z = {}
    h = {}
    g = {}
    hbar = {}

    count = 0
    for dim in layer_dims:
        z[count] = model.addVars(dim, vtype=GRB.BINARY, name="neuron_layer_" + str(count))
        h[count] = model.addVars(dim)
        hbar[count] = model.addVars(dim)
        g[count] = model.addVars(dim, lb=-GRB.INFINITY)
        count += 1

    for i in range(layer_dims[0]):
        model.addConstr(gb.quicksum(w[0][i][j] * x[j] for j in range(in_size)) + b[0][i] == g[0][i])
        model.addConstr((z[0][i] == 0) >> (h[0][i] == 0))
        model.addConstr((z[0][i] == 1) >> (h[0][i] == g[0][i]))
        model.addConstr((z[0][i] == 0) >> (g[0][i] <= 0))

    count = 1
    for input_dim, output_dim in zip(layer_dims, layer_dims[1:]):
        for i in range(output_dim):
            model.addConstr(
                gb.quicksum(w[count][i][j] * h[count - 1][j] for j in range(input_dim)) + b[count][i] == g[count][i])
            model.addConstr((z[count][i] == 0) >> (h[count][i] == 0))
            model.addConstr((z[count][i] == 1) >> (h[count][i] == g[count][i]))
            model.addConstr((z[count][i] == 0) >> (g[count][i] <= 0))
        count += 1

    model.setObjective(g[count - 1][0], GRB.MAXIMIZE)

    model.optimize(get_relaxation)

    # print(model.status)

    if model.status == 9:
        return None, None

    z_vals = gbdict2lst_z(z_vals, layer_dims)
    return x_vals, z_vals


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

    x, frac_z = get_linear_relaxation_debug(model_nn, timelimit)
    print(f'First x, {x}')
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

    print(f'first walk result: {model_nn.forward(x_max)}')

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
                print(f'walk {valid_start_count} result: {model_nn.forward(local_max)}')
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
                    print(f'walk {valid_start_count} result: {model_nn.forward(local_max)}')
            # print(f'end with 1 loop at {time.time() - start}')
            sys.stdout.flush()

    time_count = time.time() - start
    return x_max, max_, first_max, time_count, start_count, valid_start_count, update_list