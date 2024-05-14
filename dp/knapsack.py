import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

projects = [
    {"novelty": -1, "experience": 10, "cost": 2},
    {"novelty": 1, "experience": 2, "cost": 1},
    {"novelty": -0.5, "experience": 5, "cost": 1},
    {"novelty": 2, "experience": -5, "cost": 1},
]
capacity = 2


def calculate_value(project, alpha):
    return project["novelty"] * alpha + project["experience"]


def knapsack(projects, capacity, alpha):
    n = len(projects)
    # Initialize DP table
    dp = np.zeros((n + 1, capacity + 1))

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if projects[i - 1]["cost"] <= w:
                dp[i][w] = max(dp[i - 1][w],
                               dp[i - 1][w - projects[i - 1]["cost"]] + calculate_value(projects[i - 1], alpha))
            else:
                dp[i][w] = dp[i - 1][w]

    return dp[n][capacity]


# Vary alpha and collect results
alphas = np.linspace(-2, 6, 100)
values = [knapsack(projects, capacity, alpha) for alpha in alphas]

# Plot the results
plt.plot(alphas, values)
plt.xlabel("Alpha")
plt.ylabel("Total Value")
plt.title("Knapsack Value vs. Alpha")
plt.grid(True)
plt.savefig("knapsack_value_vs_alpha.png")
