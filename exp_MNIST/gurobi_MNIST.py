import numpy as np
from matplotlib import pyplot as plt
import time

import torch
import torchvision
from skorch import NeuralNetClassifier

import gurobipy as gp
from gurobipy import GRB
from gurobi_ml import add_predictor_constr

from mnist_io_csv import *
from mnist_util import *


def solve_optimal_adversary_with_gurobi(nn_regression, image, wrong_label, right_label, delta, ex_prob, timelimit, info):
    start = time.time()

    m = gp.Model()

    x = m.addMVar(image.shape, lb=0.0, ub=1.0, name="x")
    y = m.addMVar(ex_prob.detach().numpy().shape, lb=-gp.GRB.INFINITY, name="y")

    abs_diff = m.addMVar(image.shape, lb=0, ub=1, name="abs_diff")

    m.setObjective(y[wrong_label] - y[right_label], gp.GRB.MAXIMIZE)

    # Bound on the distance to example in norm-1
    m.addConstr(abs_diff >= x - image)
    m.addConstr(abs_diff >= -x + image)
    m.addConstr(abs_diff.sum() <= delta)

    pred_constr = add_predictor_constr(m, nn_regression, x, y)
    pred_constr.print_stats()

    m.setParam('TimeLimit', timelimit)
    m.setParam('OutputFlag', 1)
    m.Params.BestBdStop = 0.0
    m.Params.BestObjStop = 0.0
    m.optimize()

    time_count = time.time() - start

    # Calculate the best gap (optimality gap)
    best_gap = None
    if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        best_gap = m.MIPGap

    if m.status == 4:
        print(f'{info} is infeasible or unbounded')
        return None, None, None, time_count
    elif m.status == GRB.OPTIMAL:
        # values = [x[i].x for i in range(len(x.tolist()))]
        values = x.X.reshape(-1).tolist()
        return values, y[wrong_label].x - y[right_label].x, best_gap, time_count
    elif m.SolCount > 0:
        m.setParam(GRB.Param.SolutionNumber, 0)
        output = m.PoolObjVal
        # values = [x[i].Xn for i in range(len(x.tolist()))]
        values = x.Xn.reshape(-1).tolist()
        return values, output, best_gap, time_count
    else:
        print(f'end with status: {m.status}')
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