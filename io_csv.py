import pandas as pd
import os


def my_literal_eval(s):
    # Trim the brackets
    s = s[1:-1]

    # Split the string into components based on the comma separator
    components = s.split(',')

    # Remove leading/trailing spaces from each component, convert to float
    # And build a list from the components
    return [float(c.strip()) for c in components]


def store_data(data, filename):
    # create DataFrame
    df = pd.DataFrame(data, columns=['method', 'model_size', 'parameters', 'seed', 'x_max', 'max_', 'first_max', 'time_count', 'start_count', 'valid_start_count', 'update_list'])

    write_header = not os.path.exists(filename)

    # store into csv file (in append mode)
    df.to_csv(filename, mode='a', header=write_header, index=False)



def store_data_leaky(data, filename):
    # create DataFrame
    df = pd.DataFrame(data, columns=['method', 'model_size', 'leaky_rate', 'seed', 'leaky_max', 'x_max', 'max_', 'time_count', 'mip_gap'])

    write_header = not os.path.exists(filename)

    # store into csv file (in append mode)
    df.to_csv(filename, mode='a', header=write_header, index=False)


def store_data_leaky_callback(data, filename):
    # create DataFrame
    df = pd.DataFrame(data, columns=['method', 'model_size', 'leaky_rate', 'seed', 'status', 'x_max', 'max_', 'time_count', 'walk_count', 'mip_gap', 'update_list'])

    write_header = not os.path.exists(filename)

    # store into csv file (in append mode)
    df.to_csv(filename, mode='a', header=write_header, index=False)


# def read_data(filename):
#     # read csv file into DataFrame
#     df = pd.read_csv(filename)
#
#     # convert string representation of lists back into actual lists
#     df['x_max'] = df['x_max'].apply(my_literal_eval)
#     df['updates'] = df['updates'].apply(my_literal_eval)
#
#     return df

