import numpy as np
from matplotlib import pyplot as plt
import time

import torch
import torchvision
from skorch import NeuralNetClassifier

import gurobipy as gb
from gurobipy import GRB
from gurobi_ml import add_predictor_constr

from mnist_io_csv import *
from mnist_util import *


def solve_optimal_adversary_with_gurobi(nn_regression, image, wrong_label, right_label, delta, ex_prob, timelimit, info):
    start = time.time()

    w = get_weight_matrix(nn_regression)
    b = get_bias_matrix(nn_regression)
    in_size, layer_dims = get_model_info(nn_regression)

    model = gb.Model()
    global x_vals
    global z_vals
    x_vals = []
    z_vals = []


    model.setParam('OutputFlag', 0)
    # model.setParam('OutputFlag', 1)
    model.setParam('TimeLimit', timelimit)
    x = model.addVars(in_size, ub=1, lb=0, name="input")
    abs_diff = model.addVars(in_size, ub=1, lb=0, name="abs_diff")
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

    # Bound on the distance to example in norm-1
    # model.addConstr(abs_diff >= x - image)
    # model.addConstr(abs_diff >= -x + image)
    # model.addConstr(abs_diff.sum() <= delta)
    for i in range(in_size):
        model.addConstr(abs_diff[i] >= x[i] - image[i])
        model.addConstr(abs_diff[i] >= -x[i] + image[i])
    model.addConstr(abs_diff.sum() <= delta)

    model.setObjective(g[count - 1][wrong_label] - g[count - 1][right_label], GRB.MAXIMIZE)

    model.optimize()



    time_count = time.time() - start

    # Calculate the best gap (optimality gap)
    best_gap = None
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        best_gap = model.MIPGap

    if model.status == 4:
        print(f'{info} is infeasible or unbounded')
        return None, None, None, time_count
    elif model.status == GRB.OPTIMAL:
        values = [x[i].x for i in range(len(x))]
        # values = x.X.reshape(-1).tolist()
        return values, g[count - 1][wrong_label].x - g[count - 1][right_label].x, best_gap, time_count
    elif model.SolCount > 0:
        model.setParam(GRB.Param.SolutionNumber, 0)
        output = model.PoolObjVal
        values = [x[i].Xn for i in range(len(x))]
        # values = x.Xn.reshape(-1).tolist()
        return values, output, best_gap, time_count
    else:
        print(f'end with status: {model.status}')
        return None, None, None, time_count


def solve_with_gurobi_and_record(nn_regression, seed, image, image_index, delta, ex_prob, wrong_label, right_label, time_limit, info):
    x_max, max_, best_gap, time_count = solve_optimal_adversary_with_gurobi(nn_regression, image, wrong_label, right_label, delta, ex_prob, time_limit, info)
    input_size, layer_dims = get_model_info(nn_regression)
    layer_num = len(layer_dims) - 1
    if max_ is None:
        store_data_gurobi(
            [['GUROBI', [input_size] + layer_dims, seed, image_index, x_max, None,
              time_count,
              best_gap, None, right_label, wrong_label, info]], f'Gurobi_MNIST_Benchmark_{time_limit}_{info[-1]}.csv')
    else:
        res = nn_regression(torch.FloatTensor(x_max)).tolist()
        store_data_gurobi(
            [['GUROBI', [input_size] + layer_dims, seed, image_index, x_max, res[wrong_label] - res[right_label], time_count,
              best_gap, res, right_label, wrong_label, info]], f'Gurobi_MNIST_Benchmark_{time_limit}_{info[-1]}.csv')
    return x_max, max_, time_count