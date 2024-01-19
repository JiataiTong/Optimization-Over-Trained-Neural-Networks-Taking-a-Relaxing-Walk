import numpy as np

def set_seed_and_call_function():
    np.random.seed(42)  # Set the random seed
    return mid_call()

def mid_call():
    return call_random_choice()

def call_random_choice():
    return np.random.choice([0, 1])

# Call the function
result = set_seed_and_call_function()
print(result)