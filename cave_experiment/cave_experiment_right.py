import torch
import numpy as np

from matplotlib import pyplot as plt
from pyepo.data import dataset
from pyepo.model.grb.knapsack import knapsackModel
from torch.utils.data import DataLoader

from CaVEmain.src.cave import exactConeAlignedCosine
from CaVEmain.src.dataset import optDatasetConstrs
from data_generator import generate_data
from dp.dynamic import DP_Knapsack
from predmodel import ValueModel

right_data = np.load("../labled_data/right_labled_data.npy", allow_pickle=True)
num_runs = len(right_data)
tf_runs = []
cave_alphas = []
dp_alphas = []
alpha_values = np.arange(right_data[0]['alpha'][0], right_data[0]['alpha'][1], 0.05)
num_items = right_data[0]['num_items']
capacity = right_data[0]['capacity']
for i in range(num_runs):
    if i % 10 == 0:
        print(f"Running iteration: {i}")
    torch.manual_seed(right_data[i]['seed'])

    weights, features, values = generate_data(num_items=num_items, capacity=capacity, seed=right_data[i]['seed'])

    optmodel = knapsackModel(weights=weights, capacity=capacity)
    data = optDatasetConstrs(model=optmodel, feats=features, costs=values)
    dataloader = DataLoader(data, batch_size=1, shuffle=True)

    cave = exactConeAlignedCosine(optmodel=optmodel, solver="clarabel")

    # Estimate gradients with dynamic programming
    features = features.reshape((2, num_items))
    dp_model = DP_Knapsack(
        weights[0], features, values, capacity, alpha_values[0], alpha_values[-1]
    )
    dp_model.solve()

    # Estimate gradients with loss functions
    cave_values = []
    cave_gradients = []

    for data in dataloader:
        x, c, w, z, bctr = data
        x = torch.reshape(x, (2, num_items))

        for alpha in alpha_values:
            predmodel = ValueModel(alpha=alpha)
            cp = predmodel.forward(x)

            cave_loss = cave(cp, bctr)
            cave_loss.backward(retain_graph=True)
            cave_values.append(cave_loss.item())
            cave_gradients.append(predmodel.alpha.grad.item())

            predmodel.zero_grad()

    # # Plot loss function gradients
    # # Create base plot with DP solutions
    _, horizontal_plots, _, intervals = dp_model.plot(linear=False, horizontal=True)
    # plt.grid(True)
    # plt.xlabel("Alpha")
    # plt.ylabel("Gradient")

    # Plot CaVE on top
    cave_grad_plot = plt.plot(alpha_values, cave_gradients, color="green")
    # plt.title("CaVE loss gradient vs. alpha")
    # plt.legend(
    #     [horizontal_plots, cave_grad_plot[0]],
    #     ["DP", "CAVE"],
    #     handler_map={tuple: HandlerTuple(ndivide=None)},
    # )

    cave_alpha = None
    for j in range(len(alpha_values) - 1):
        if cave_grad_plot[0].get_ydata()[j] <= 0 <= cave_grad_plot[0].get_ydata()[j + 1]:
            cave_alpha = (cave_grad_plot[0].get_xdata()[j] + cave_grad_plot[0].get_xdata()[j + 1]) / 2
            break

    max_dp_value = 0
    max_dp_range = []
    dp_alpha = None
    for hor_plot, interval in zip(horizontal_plots, intervals):
        cur_dp = hor_plot[0]
        if cur_dp > max_dp_value:
            max_dp_value = cur_dp
            max_dp_range = interval
            dp_alpha = (max_dp_range[0] + max_dp_range[-1]) / 2

    epsilon = 0.2
    if cave_alpha is not None and dp_alpha is not None:
        tf_runs.append(max_dp_range[0] - epsilon < cave_alpha < max_dp_range[1] + epsilon)
        cave_alphas.append(cave_alpha)
        dp_alphas.append(dp_alpha)

        # plt.title("CaVE loss gradient vs. alpha")
        # plt.savefig("cave_experiment.png")
        # # [hor_plot.remove() for hor_plot in horizontal_plots]
        # # cave_grad_plot[0].remove()
        # plt.clf()


def get_histogram_data(alphas, tf_runs, bins):
    true_counts = np.zeros(len(bins) - 1)
    false_counts = np.zeros(len(bins) - 1)

    for i, bin_edge in enumerate(bins[:-1]):
        in_bin = (alphas >= bins[i]) & (alphas < bins[i + 1])
        true_counts[i] = np.sum(tf_runs[in_bin])
        false_counts[i] = np.sum(~tf_runs[in_bin])

    return true_counts, false_counts


width = 0.3
bins = np.arange(right_data[0]['alpha'][0], right_data[0]['alpha'][1], width)
true_counts_cave, false_counts_cave = get_histogram_data(cave_alphas, np.array(tf_runs), bins)
true_counts_dp, false_counts_dp = get_histogram_data(dp_alphas, np.array(tf_runs), bins)

fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Plot for cave_alphas
ax[0].bar(bins[:-1], true_counts_cave, width=width, align='edge', label='True')
ax[0].bar(bins[:-1], false_counts_cave, width=width, align='edge', bottom=true_counts_cave, label='False')
ax[0].set_title('cave_alphas')
ax[0].set_ylabel('Count')
ax[0].legend()

# Plot for dp_alphas
ax[1].bar(bins[:-1], true_counts_dp, width=width, align='edge', label='True')
ax[1].bar(bins[:-1], false_counts_dp, width=width, align='edge', bottom=true_counts_dp, label='False')
ax[1].set_title('dp_alphas')
ax[1].set_xlabel('Alpha')
ax[1].set_ylabel('Count')
ax[1].legend()

plt.savefig('cave_experiments_right.png')
