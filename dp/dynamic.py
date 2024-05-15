import pyepo
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colormaps


# generate data for 2D knapsack
m = 4 # number of items
n = 1 # number of data????????????????????????????
p = 2 * m # size of feature
deg = 6 # polynomial degree
dim = 1 # dimension of knapsack
noise_width = 0.5 # noise half-width
caps = [20] * dim # capacity
weights, x, c = pyepo.data.knapsack.genData(n, p, m, deg=deg, dim=dim, noise_width=noise_width, seed=32)
x = x.reshape((m, -1))

capacity = caps[0]


def calculate_value(x, alpha):
    return x[0] * alpha + x[1]


def knapsack(weights, x, c, alpha):
    n = x.shape[0]
    # Initialize DP table
    dp = np.zeros((n + 1, capacity + 1))
    val = np.zeros((n + 1, capacity + 1))
    int_weights = np.ceil(weights).astype('int')
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w],
                               dp[i - 1][w - int_weights[i - 1]] + calculate_value(x[i - 1], alpha))
                if dp[i - 1][w] > dp[i - 1][w - int_weights[i - 1]] + calculate_value(x[i - 1], alpha):
                    val[i][w] = val[i - 1][w]
                else:
                    val[i][w] = val[i - 1][w - int_weights[i - 1]] + c[i - 1]
            else:
                dp[i][w] = dp[i - 1][w]
                val[i][w] = val[i - 1][w]

    return dp[n][capacity], val[n][capacity]


# Vary alpha and collect results
alphas = np.linspace(-2, 6, 1000)
values = [knapsack(weights[0], x, c[0], alpha)[0] for alpha in alphas]
true_values = [knapsack(weights[0], x, c[0], alpha)[1] for alpha in alphas]
# Plot the results
plt.plot(alphas, values)
plt.plot(alphas, true_values)
plt.xlabel("Alpha")
plt.ylabel("Total Value")
plt.title("Knapsack Value vs. Alpha")
plt.grid(True)
plt.savefig("knapsack_value_vs_alpha.png")