from main import *
from gurobi_benchmark import *

seed_list = [16, 17, 18, 19, 20]
input_size_list = [10, 100, 1000]
layer_num_list = [1, 2]
layer_size_list = [100, 500]
time_limits = [300, 3600]


gr_df_exist = True


if os.path.exists(f'Gurobi_Benchmark.csv'):
    gr_df = pd.read_csv(f'Gurobi_Benchmark.csv')
else:
    gr_df_exist = False
    gr_df = pd.DataFrame()
for time_limit in time_limits:
    for input_size in input_size_list:
        for layer_num in layer_num_list:
            for layer_size in layer_size_list:
                for seed in seed_list:
                    tag = str([input_size] + layer_num * [layer_size] + [1])
                    # print(tag)
                    print(
                        f'[{input_size}, {layer_num} x {layer_size}, 1] with seed {seed} Gurobi MIP')
                    inHistory = False
                    if gr_df_exist and gr_df['model_size'].isin([tag]).any():
                        results = gr_df.index[gr_df['model_size'] == tag].tolist()
                        for result in results:
                            if gr_df.loc[result, 'seed'] == seed:
                                max_ = gr_df.loc[result, 'max_']
                                time_count = gr_df.loc[result, 'time_count']
                                print(f'max: {max_}')
                                print(f'time: {time_count}')
                                print(f'-----------------------------------------')
                                inHistory = True
                                break
                        if not inHistory:
                            result = solve_with_gurobi(input_size, layer_num, layer_size, seed, time_limit)
                            print(f'max: {result[1]}')
                            print(f'time: {result[2]}')
                            print(f'-----------------------------------------')
                    else:
                        result = solve_with_gurobi(input_size, layer_num, layer_size, seed, time_limit)
                        print(f'max: {result[1]}')
                        print(f'time: {result[2]}')
                        print(f'-----------------------------------------')




