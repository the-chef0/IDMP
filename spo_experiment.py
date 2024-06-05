import torch
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from pyepo.data import dataset
from pyepo.func import SPOPlus
from pyepo.model.grb.knapsack import knapsackModel
from torch.utils.data import DataLoader
from data_generator import generate_data
from dp.dynamic import DP_Knapsack
from predmodel import ValueModel

num_runs = 50
tf_runs = []
alphas = []
alpha_values = np.arange(-7, 7, 0.05)
num_items = 10
capacity = 20
for i in range(num_runs):
    torch.manual_seed(i)

    weights, features, values = generate_data(num_items=num_items, capacity=capacity)

    optmodel = knapsackModel(weights=weights, capacity=capacity)
    data = dataset.optDataset(model=optmodel, feats=features, costs=values)
    dataloader = DataLoader(data, batch_size=1, shuffle=True)

    spop = SPOPlus(optmodel=optmodel)

    # Estimate gradients with dynamic programming
    features = features.reshape((2, num_items))
    dp_model = DP_Knapsack(
        weights[0], features, values, capacity, alpha_values[0], alpha_values[-1]
    )
    dp_model.solve()

    # Estimate gradients with loss functions
    spop_values = []
    spop_gradients = []

    for data in dataloader:
        x, c, w, z = data
        x = torch.reshape(x, (2, num_items))

        for alpha in alpha_values:
            predmodel = ValueModel(alpha=alpha)
            cp = predmodel.forward(x)

            spop_loss = spop(cp, c, w, z)
            spop_loss.backward(retain_graph=True)
            spop_values.append(spop_loss.item())
            spop_gradients.append(predmodel.alpha.grad.item())

            predmodel.zero_grad()

    # Plot loss function gradients
    # Create base plot with DP solutions
    _, horizontal_plots = dp_model.plot(linear=False, horizontal=True)
    plt.grid(True)
    plt.xlabel("Alpha")
    plt.ylabel("Gradient")

    # Plot SPO on top
    spop_grad_plot = plt.plot(alpha_values, spop_gradients, color="green")
    plt.title("SPO+ loss gradient vs. alpha")
    plt.legend(
        [horizontal_plots, spop_grad_plot[0]],
        ["DP", "SPO+"],
        handler_map={tuple: HandlerTuple(ndivide=None)},
    )

    max_jump = 0
    max_alpha = None
    for j in range(len(alpha_values) - 1):
        cur_jump = abs(spop_grad_plot[0].get_ydata()[j] - spop_grad_plot[0].get_ydata()[j + 1])
        if cur_jump > max_jump:
            max_jump = cur_jump
            max_alpha = spop_grad_plot[0].get_xdata()[j]

    max_dp_value = 0
    max_dp_range = []
    for hor_plot in horizontal_plots:
        cur_dp = hor_plot.get_ydata()[0]
        if cur_dp > max_dp_value:
            max_dp_value = cur_dp
            max_dp_range = hor_plot.get_xdata()

    tf_runs.append(max_dp_range[0] < max_alpha < max_dp_range[1])
    alphas.append(max_alpha)

    plt.title("SPO+ loss gradient vs. alpha")
    plt.savefig("spo_experiment.png")
    # [hor_plot.remove() for hor_plot in horizontal_plots]
    # spop_grad_plot[0].remove()
    plt.clf()

unique_alphas = sorted(set(alphas))
true_counts = [sum(1 for alpha, result in zip(alphas, tf_runs) if alpha == a and result) for a in unique_alphas]
false_counts = [sum(1 for alpha, result in zip(alphas, tf_runs) if alpha == a and not result) for a in unique_alphas]

# Plotting
x = range(len(unique_alphas))
width = 1

plt.bar(x, true_counts, width, label='True', color='blue')
plt.bar([i + width for i in x], false_counts, width, label='False', color='red')

plt.xlabel('Alpha')
plt.ylabel('Counts')
plt.title('Experiment results by alpha')
plt.xticks([i + width / 2 for i in x], unique_alphas)
plt.legend()

plt.savefig('spo_experiments.png')
