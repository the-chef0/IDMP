import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

projects = [
    {"novelty": -1, "experience": 10, "cost": 2, "value": 14},
    {"novelty": 1, "experience": 2, "cost": 1, "value": 11},
    {"novelty": -0.5, "experience": 5, "cost": 1, "value": 12},
    {"novelty": 2, "experience": -5, "cost": 1, "value": 10},
]

capacity = 2


def calculate_value(project, alpha):
    return project["novelty"] * alpha + project["experience"]


def knapsack(projects, capacity, alpha):
    n = len(projects)
    # Initialize DP table
    dp = np.zeros((n + 1, capacity + 1))
    val = np.zeros((n + 1, capacity + 1))

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if projects[i - 1]["cost"] <= w:
                dp[i][w] = max(dp[i - 1][w],
                               dp[i - 1][w - projects[i - 1]["cost"]] + calculate_value(projects[i - 1], alpha))
                if dp[i - 1][w] > dp[i - 1][w - projects[i - 1]["cost"]] + calculate_value(projects[i - 1], alpha):
                    val[i][w] = val[i - 1][w]
                else:
                    val[i][w] = val[i - 1][w - projects[i - 1]["cost"]] + projects[i - 1]["value"]
            else:
                dp[i][w] = dp[i - 1][w]
                val[i][w] = val[i - 1][w]

    return dp[n][capacity], val[n][capacity]


# Vary alpha and collect results
alphas = np.linspace(-2, 6, 100)
values = [knapsack(projects, capacity, alpha)[0] for alpha in alphas]
true_values = [knapsack(projects, capacity, alpha)[1] for alpha in alphas]
# Plot the results
plt.plot(alphas, values)
plt.plot(alphas, true_values)
plt.xlabel("Alpha")
plt.ylabel("Total Value")
plt.title("Knapsack Value vs. Alpha")
plt.grid(True)
plt.savefig("knapsack_value_vs_alpha.png")
