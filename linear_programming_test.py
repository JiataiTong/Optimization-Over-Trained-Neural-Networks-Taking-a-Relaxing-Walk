import gurobipy as gb
from gurobipy import GRB
from util import *
import numpy as np
from Network import Network
import random

def solve_lp_original(modelnn, activation_pattern):
    w = modelnn.get_weight_matrix()
    b = modelnn.get_bias_matrix()
    layer_dims = modelnn.layer_dims
    in_size = modelnn.in_size

    model = gb.Model()
    model.setParam('OutputFlag', 0)
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
        model.addConstr(z[0][i] == activation_pattern[0][i])

    count = 1
    for input_dim, output_dim in zip(layer_dims, layer_dims[1:]):
        for i in range(output_dim):
            model.addConstr(
                gb.quicksum(w[count][i][j] * h[count - 1][j] for j in range(input_dim)) + b[count][i] == g[count][i])
            model.addConstr((z[count][i] == 0) >> (h[count][i] == 0))
            model.addConstr((z[count][i] == 1) >> (h[count][i] == g[count][i]))
            model.addConstr((z[count][i] == 0) >> (g[count][i] <= 0))
            model.addConstr(z[count][i] == activation_pattern[count][i])
        count += 1

    model.setObjective(g[count - 1][0], GRB.MAXIMIZE)
    model.optimize()

    max_lp = model.getAttr('ObjVal')
    x_new = gbdict2lst(x, len(x))
    g = gbdict2lst(g, layer_dims)
    return max_lp, x_new, g


def solve_lp_M(modelnn, activation_pattern):
    BIG_M = 5000.0
    EPSILON = 1e-08
    w = modelnn.get_weight_matrix()
    b = modelnn.get_bias_matrix()
    layer_dims = modelnn.layer_dims
    in_size = modelnn.in_size

    model = gb.Model()
    model.setParam('OutputFlag', 0)
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

        model.addConstr(h[0][i] <= BIG_M * z[0][i] + EPSILON)
        model.addConstr(h[0][i] >= 0.0)

        model.addConstr(g[0][i] <= BIG_M * z[0][i])
        model.addConstr(g[0][i] >= -1.0 * BIG_M * (1 - z[0][i]) + EPSILON)

        model.addConstr(h[0][i] <= g[0][i] + BIG_M * (1 - z[0][i]) + EPSILON)
        model.addConstr(h[0][i] >= g[0][i] - BIG_M * (1 - z[0][i]) - EPSILON)
        model.addConstr(z[0][i] == activation_pattern[0][i])

    count = 1
    for input_dim, output_dim in zip(layer_dims, layer_dims[1:]):
        for i in range(output_dim):
            model.addConstr(
                gb.quicksum(w[count][i][j] * h[count - 1][j] for j in range(input_dim)) + b[count][i] == g[count][i])

            model.addConstr(h[count][i] <= BIG_M * z[count][i] + EPSILON)
            model.addConstr(h[count][i] >= 0.0)

            model.addConstr(g[count][i] <= BIG_M * z[count][i])
            model.addConstr(g[count][i] >= -1.0 * BIG_M * (1 - z[count][i]) + EPSILON)

            model.addConstr(
                h[count][i] <= g[count][i] + BIG_M * (1 - z[count][i]) + EPSILON)
            model.addConstr(
                h[count][i] >= g[count][i] - BIG_M * (1 - z[count][i]) - EPSILON)

        model.addConstr(z[count][i] == activation_pattern[count][i])
        count += 1

    model.setObjective(g[count - 1][0], GRB.MAXIMIZE)
    model.optimize()

    max_lp = model.getAttr('ObjVal')
    x_new = gbdict2lst(x, len(x))
    g = gbdict2lst(g, layer_dims)
    return max_lp, x_new, g


def calculate_coeffs_constants(weight, bias, ap):
    num_layers = len(weight)
    # print(num_layers)

    # Initialize dictionaries for coefficients, constants, pre-activate coefficients and constants.
    coeffs = {}
    constants = {}
    pre_coeffs = {}
    pre_constants = {}

    coeffs[-1] = np.identity(weight[0].shape[1])
    constants[-1] = np.zeros(weight[0].shape[1])
    pre_coeffs[-1] = coeffs[-1].copy()
    pre_constants[-1] = constants[-1].copy()

    for l in range(num_layers):
        layer_size = len(ap[l])
        prev_layer_size = coeffs[l - 1].shape[1]

        coeffs[l] = np.zeros((layer_size, prev_layer_size))
        constants[l] = np.zeros(layer_size)
        pre_coeffs[l] = np.zeros((layer_size, prev_layer_size))
        pre_constants[l] = np.zeros(layer_size)

        for j in range(layer_size):
            # for i in range(prev_layer_size):
            #     pre_coeffs[l][j, i] += np.dot(weight[l][j, :], coeffs[l - 1][:, i])
            pre_coeffs[l][j, :] = weight[l][j, :] @ coeffs[l - 1]
            pre_constants[l][j] += np.dot(weight[l][j, :], constants[l - 1]) + bias[l][j]

            if ap[l][j] == 0:
                continue

            coeffs[l][j] = pre_coeffs[l][j]
            constants[l][j] = pre_constants[l][j]

    return pre_coeffs, pre_constants


def solve_lp_pre_calc(modelnn, activation_pattern):
    w = modelnn.get_weight_matrix()
    b = modelnn.get_bias_matrix()
    layer_dims = modelnn.layer_dims
    in_size = modelnn.in_size

    coeffs, constants = calculate_coeffs_constants(w, b, activation_pattern)

    model = gb.Model()
    model.setParam('OutputFlag', 0)
    x = model.addVars(in_size, ub=1, lb=0, name="input")
    g = {}
    count = 0
    for dim in layer_dims:
        g[count] = model.addVars(dim, lb=-GRB.INFINITY)
        count += 1

    # print(len(coeffs), len(coeffs[len(w) - 1]))

    constraints_count = 0

    count = 0

    for output_dim in layer_dims[:-1]:
        # if count == len(w) - 1:
        #     print("here")
        #     break
        # print(output_dim, count)
        for i in range(output_dim):
            model.addConstr(
                g[count][i] == gb.quicksum(coeffs[count][i][j] * x[j] for j in range(in_size)) + constants[count][i])
            if activation_pattern[count][i] == 1:
                # model.addConstr(
                #     gb.quicksum(coeffs[count][i][j] * x[j] for j in range(in_size)) + constants[count][i] >= 0)
                model.addConstr(g[count][i] >= 0)
            else:
                # model.addConstr(
                #     gb.quicksum(coeffs[count][i][j] * x[j] for j in range(in_size)) + constants[count][i] <= 0)
                model.addConstr(g[count][i] <= 0)
            constraints_count += 1
        count += 1

    # print("constraints count: ", constraints_count)

    model.setObjective(gb.quicksum(coeffs[len(w) - 1][0][j] * x[j] for j in range(in_size)) + constants[len(w) - 1][0], GRB.MAXIMIZE)
    # model.setParam('Method', 1)
    model.optimize()

    print("status: ", model.status)
    # model.computeIIS()
    # model.write("model.ilp")

    max_lp = model.getAttr('ObjVal')
    x_new = gbdict2lst(x, len(x))
    g = gbdict2lst(g, layer_dims)
    return max_lp, x_new, g


input_size = 1000
layer_dims = [500, 500, 1]
seed_list = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

for seed in seed_list:
    model_nn = Network(in_size=input_size, layer_dims=layer_dims, seed=seed)

    for i in range(10):
        x = np.random.rand(model_nn.in_size)


        print(f'---seed = {seed} ---')

        ap = get_binary_activations(model_nn, x)

        ts = time.time()
        lp, x, _ = solve_lp_pre_calc(model_nn, ap)
        cost = time.time() - ts
        # print(x)

        print(f'pre_calc: lp = {lp}, network_ret = {model_nn(torch.FloatTensor(x)).item()} , time = {cost}')


        ts = time.time()
        lp, x, _ = solve_lp_original(model_nn, ap)
        cost = time.time() - ts
        print(f'original: lp = {lp}, network_ret = {model_nn(torch.FloatTensor(x)).item()}, time = {cost}')






        # ts = time.time()
        # lp, _, _ = solve_lp_M(model_nn, ap)
        # cost = time.time() - ts
        # print(f'big M: lp = {lp}, time = {cost}')






