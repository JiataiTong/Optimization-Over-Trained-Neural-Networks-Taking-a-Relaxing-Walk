import gurobipy as gb
from gurobipy import GRB
import numpy as np
from mnist_util import *


def calculate_coeffs_constants(weight, bias, ap):
    num_layers = len(weight)
    # print(len(ap))

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
        # print(layer_size)
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


def solve_lp_pre_calc_with_g(modelnn, image, wrong_label, right_label, activation_pattern, delta):
    w = get_weight_matrix(modelnn)
    b = get_bias_matrix(modelnn)
    in_size, layer_dims = get_model_info(modelnn)

    coeffs, constants = calculate_coeffs_constants(w, b, activation_pattern)

    model = gb.Model()
    model.setParam('OutputFlag', 1)
    x = model.addVars(in_size, ub=1, lb=0, name="input")
    abs_diff = model.addVars(in_size, ub=1, lb=0, name="abs_diff")
    g = {}
    count = 0
    for dim in layer_dims:
        g[count] = model.addVars(dim, lb=-GRB.INFINITY)
        count += 1

    # print(len(coeffs), len(coeffs[len(w) - 1]))

    constraints_count = 0

    count = 0

    for output_dim in layer_dims[:-1]:
        for i in range(output_dim):
            model.addConstr(
                g[count][i] == gb.quicksum(coeffs[count][i][j] * x[j] for j in range(in_size)) + constants[count][i])
            if activation_pattern[count][i] == 1:
                model.addConstr(g[count][i] >= 0)
            else:
                model.addConstr(g[count][i] <= 0)
            constraints_count += 1
        count += 1

    for i in range(layer_dims[-1]):
        model.addConstr(g[count][i] == gb.quicksum(coeffs[count][i][j] * x[j] for j in range(in_size)) + constants[count][i])

    print(count)

    # Extra Constranints
    # Bound on the distance to example in norm-1
    # model.addConstr(abs_diff >= x - image)
    # model.addConstr(abs_diff >= -x + image)
    # model.addConstr(abs_diff.sum() <= delta)
    for i in range(in_size):
        model.addConstr(abs_diff[i] >= x[i] - image[i])
        model.addConstr(abs_diff[i] >= -x[i] + image[i])
    model.addConstr(abs_diff.sum() <= delta)

    model.setObjective(g[count][wrong_label] - g[count][right_label], gb.GRB.MAXIMIZE)

    model.optimize()

    max_lp = model.getAttr('ObjVal')
    x_new = gbdict2lst(x, len(x))
    g = gbdict2lst(g, layer_dims)
    return max_lp, x_new, g



def solve_lp_with_g(modelnn, image, wrong_label, right_label, activation_pattern, delta):
    w = get_weight_matrix(modelnn)
    b = get_bias_matrix(modelnn)
    in_size, layer_dims = get_model_info(modelnn)

    model = gb.Model()
    model.setParam('OutputFlag', 0)
    x = model.addVars(in_size, ub=1, lb=0, name="input")
    abs_diff = model.addVars(in_size, ub=1, lb=0, name="abs_diff")
    z = {}
    h = {}
    g = {}

    count = 0
    for dim in layer_dims:
        g[count] = model.addVars(dim, lb=-GRB.INFINITY)
        count += 1

    # print(len(coeffs), len(coeffs[len(w) - 1]))

    count = 0
    for dim in layer_dims:
        z[count] = model.addVars(dim, vtype=GRB.BINARY, name="neuron_layer_" + str(count))
        h[count] = model.addVars(dim)
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
            # if count == len(layer_dims) - 1:
            #     continue
            model.addConstr(z[count][i] == activation_pattern[count][i])
        count += 1

    # Extra Constranints
    # Bound on the distance to example in norm-1
    # model.addConstr(abs_diff >= x - image)
    # model.addConstr(abs_diff >= -x + image)
    # model.addConstr(abs_diff.sum() <= delta)
    for i in range(in_size):
        model.addConstr(abs_diff[i] >= x[i] - image[i])
        model.addConstr(abs_diff[i] >= -x[i] + image[i])
    model.addConstr(abs_diff.sum() <= delta)

    model.setObjective(g[count - 1][wrong_label] - g[count - 1][right_label], gb.GRB.MAXIMIZE)

    model.optimize()

    max_lp = model.getAttr('ObjVal')
    x_new = gbdict2lst(x, len(x))
    g = gbdict2lst(g, layer_dims)
    return max_lp, x_new, g





