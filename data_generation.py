import random
import torch
import numpy as np

# fix random seed
random.seed(100)
np.random.seed(100)
torch.manual_seed(100)

# generate data for 1D knapsack
m = 100 # number of items
n = 1000 # number of data AKA number of ALPHAs
p = 1 # size of feature
deg = 1 # polynomial degree
dim = 1 # dimension of knapsack

weights = np.array([np.round(np.random.uniform(1, 3, m),2)])
caps = [40] # capacity
ni = np.random.uniform(-1, 1, m)
ei = np.random.uniform(-1, 1, m)
noise = 0.5 
alpha = np.random.uniform(-10, 10, n)

real_values = []
for j in range(n):
    data_point = []
    for i in range(m):
        data_point.append(np.round(np.random.normal(ni[i]*alpha[j] + ei[i], noise),0))   # Cost of the items
        #data_point.append(np.round(np.random.normal(ni[i]*ni[i]*alpha[j] + ei[i], noise),0))   # Cost of the items
    real_values.append(data_point)     

x = np.array(alpha).reshape((n, 1)) # Parameters, in our case Alpha
c = np.array(real_values)       # Cost of the items