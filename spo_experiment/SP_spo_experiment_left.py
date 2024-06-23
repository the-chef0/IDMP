import torch
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from pyepo.data import dataset
from torch.utils.data import DataLoader
from dp.SP_dynamic import SP_dynamic
from dp.dynamic import DP_Knapsack
from predmodel import ValueModel
from pyepo.model.grb.shortestpath import shortestPathModel
import pyepo.data.shortestpath
from pyepo.func import SPOPlus

left_data = np.load("labled_data/SP_left_labled_data.npy", allow_pickle=True)
# num_runs = len(left_data)
num_runs = 100
tf_runs = []
spop_alphas = []
dp_alphas = []
alpha_values = np.arange(left_data[0]["alpha"][0], left_data[0]["alpha"][1], 0.05)
num_data = 1
grid = left_data[0]["grid_size"]
num_feat = 2 * ((grid[0] - 1) * grid[1] + (grid[1] - 1) * grid[0])
lable = left_data[0]["label"]
for i in range(num_runs):
    if i % 10 == 0:
        print(f"Running iteration: {i}")
    seed = left_data[i]["seed"]
    torch.manual_seed(seed)
    x, c = pyepo.data.shortestpath.genData(
        num_data, num_feat, grid, deg=1, noise_width=0, seed=seed
    )
    optmodel = shortestPathModel(grid=grid)
    data = dataset.optDataset(model=optmodel, feats=x, costs=c)
    dataloader = DataLoader(data, batch_size=1, shuffle=True)

    spop = SPOPlus(optmodel=optmodel)

    x = x.reshape((2, -1))
    sp_dynamic = SP_dynamic(
        x, c, grid, left_data[0]["alpha"][0], left_data[0]["alpha"][1]
    )
    sp_dynamic.solve()

    # Estimate gradients with loss functions
    spop_values = []
    spop_gradients = []

    for data in dataloader:
        x, c, w, z = data
        x = torch.reshape(x, (2, -1))

        for alpha in alpha_values:
            predmodel = ValueModel(alpha=alpha)
            cp = predmodel.forward(x)

            spop_loss = spop(cp, c, w, z)
            spop_loss.backward(retain_graph=True)
            spop_values.append(spop_loss.item())
            spop_gradients.append(predmodel.alpha.grad.item())

            predmodel.zero_grad()

    # # Plot loss function gradients
    # # Create base plot with DP solutions
    _, horizontal_plots, _, intervals = sp_dynamic.plot(linear=False, horizontal=True)
    # plt.grid(True)
    # plt.xlabel("Alpha")
    # plt.ylabel("Gradient")

    # Plot SPO on top
    spop_grad_plot = plt.plot(alpha_values, spop_gradients, color="green")
    # plt.title("SPO+ loss gradient vs. alpha")
    # plt.legend(
    #     [horizontal_plots, spop_grad_plot[0]],
    #     ["DP", "SPO+"],
    #     handler_map={tuple: HandlerTuple(ndivide=None)},
    # )

    spop_alpha = None
    for j in range(len(alpha_values) - 1):
        if (
            spop_grad_plot[0].get_ydata()[j]
            <= 0
            <= spop_grad_plot[0].get_ydata()[j + 1]
        ):
            spop_alpha = (
                spop_grad_plot[0].get_xdata()[j] + spop_grad_plot[0].get_xdata()[j + 1]
            ) / 2
            break

    min_dp_value = np.inf
    min_dp_range = []
    dp_alpha = None
    for hor_plot, interval in zip(horizontal_plots, intervals):
        cur_dp = hor_plot[0]
        if cur_dp < min_dp_value:
            min_dp_value = cur_dp
            min_dp_range = interval
            dp_alpha = (min_dp_range[0] + min_dp_range[-1]) / 2

    epsilon = 0.2
    if spop_alpha is not None and dp_alpha is not None:
        tf_runs.append(
            min_dp_range[0] - epsilon < spop_alpha < min_dp_range[1] + epsilon
        )
        spop_alphas.append(spop_alpha)
        dp_alphas.append(dp_alpha)
        # plt.title("SPO+ loss gradient vs. alpha")
        # plt.savefig("spo_experiment.png")
        # # [hor_plot.remove() for hor_plot in horizontal_plots]
        # # spop_grad_plot[0].remove()
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
bins = np.arange(left_data[0]["alpha"][0], left_data[0]["alpha"][1], width)
true_counts_spop, false_counts_spop = get_histogram_data(
    spop_alphas, np.array(tf_runs), bins
)
true_counts_dp, false_counts_dp = get_histogram_data(dp_alphas, np.array(tf_runs), bins)

fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Plot for spop_alphas
ax[0].bar(bins[:-1], true_counts_spop, width=width, align="edge", label="True")
ax[0].bar(
    bins[:-1],
    false_counts_spop,
    width=width,
    align="edge",
    bottom=true_counts_spop,
    label="False",
)
ax[0].set_title("spop_alphas")
ax[0].set_ylabel("Count")
ax[0].legend()

# Plot for dp_alphas
ax[1].bar(bins[:-1], true_counts_dp, width=width, align="edge", label="True")
ax[1].bar(
    bins[:-1],
    false_counts_dp,
    width=width,
    align="edge",
    bottom=true_counts_dp,
    label="False",
)
ax[1].set_title("dp_alphas")
ax[1].set_xlabel("Alpha")
ax[1].set_ylabel("Count")
ax[1].legend()

plt.savefig("SP_spo_experiments_left.png")
