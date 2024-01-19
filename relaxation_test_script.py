import pandas as pd
from gurobi_benchmark import *
from relaxation_test import *

seed_list = [50, 51, 52, 53, 54]
input_size_list = [10, 100, 1000]
layer_num_list = [2, 3]
layer_size_list = [100, 500]
# timelimit = 3600 * 2
timelimit = 600

walk_eps = 0.01
pick_bias = 0.05
gap = 2/3




for input_size in input_size_list:
    for layer_num in layer_num_list:
        for layer_size in layer_size_list:
            for seed in seed_list:
                tag = str([input_size] + layer_num * [layer_size] + [1])
                print(f'[{input_size}, {layer_num} x {layer_size}, 1] with seed {seed} relaxation test')
                inHistory = False
                result = relaxation_test(input_size, layer_num, layer_size, seed, walk_eps, pick_bias,
                                            timelimit)
                if len(result) == 0:
                    print(print(f'time_average: >{timelimit}'))
                else:
                    result = np.array(result)
                    result_nan = np.where(result == None, np.nan, result)
                    print(f'time_average: {np.nanmean(result_nan)}')
                print(f'-----------------------------------------')
