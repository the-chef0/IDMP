import numpy as np

# The weights are random between 3 and 8
# The alpha values are sampled uniformly between -10 and 10



def generate_data(
    num_items: int, data_points: int = 1, num_features: int = 1, capacity: int = 20,
seed = 42):
    np.random.seed(seed)
    # generate data for 1D knapsack
    m = num_items  # m number of items
    n = data_points  # n number of data AKA number of ALPHAs
    # p = num_features # p number of features (We are optimizing over alpha)
    deg = 1  # polynomial degree
    dim = 1  # dimension of knapsack

    weights = np.array([np.round(np.random.uniform(3, 8, m), 2)])
    caps = [capacity]  # capacity
    ni = np.random.uniform(-10, 10, m)
    ei = np.random.uniform(-10, 10, m)
    noise = 0.5
    alpha = np.random.uniform(-10, 10, n)

    real_values = []
    for j in range(n):
        data_point = []
        for i in range(m):
            # data_point.append(np.round(np.random.normal(ni[i]*alpha[j] + ei[i], noise),0))   # Cost of the items
            noise = np.random.uniform(1, 2)
            value = np.max(np.sqrt((ni[i] * alpha[j]) ** 2 + ei[i] ** 2 + noise), 0)
            value = np.round(value, 0)

            #Periodic True Values
            #value = np.round(np.random.normal(ni[i]*7 + ei[i], 0),0)
            #value = np.round(np.random.normal(ni[i]*(np.cos(alpha[j])+1) + ei[i], noise),0)
            #value = np.round(np.random.normal(ni[i]*np.absolute(alpha[j]) + ei[i], noise),0)
            #value = np.round(np.random.normal(np.random.uniform(1, 10), noise),0)
            #value = np.round(value, 0)

            data_point.append(value)
            # data_point.append(np.round(np.random.normal(ni[i]*ni[i]*alpha[j] + ei[i], noise),0))   # Cost of the items
        real_values.append(data_point)

    # features = np.array(alpha).reshape((n, 1)) # Parameters, in our case Alpha
    features = np.concatenate((ni, ei), axis=0)
    features = np.expand_dims(features, axis=0)
    values = np.array(real_values)  # Cost of the items

    return weights, features, values
