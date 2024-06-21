import torch
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from pyepo.data import dataset
from pyepo.func import perturbedFenchelYoung
from pyepo.model.grb.knapsack import knapsackModel
from torch.utils.data import DataLoader
from dp.SP_dynamic import SP_dynamic
from dp.dynamic import DP_Knapsack
from predmodel import ValueModel
from pyepo.model.grb.shortestpath import shortestPathModel
import pyepo.data.shortestpath

left_data = np.load("labled_data/SP_left_labled_data.npy", allow_pickle=True)
# num_runs = len(left_data)
num_runs = 20
tf_runs = []
pfyl_alphas = []
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

    pfyl = perturbedFenchelYoung(optmodel=optmodel)

    x = x.reshape((2, -1))
    sp_dynamic = SP_dynamic(x, c, grid, -5, 5)
    sp_dynamic.solve()

    # Estimate gradients with loss functions
    pfyl_values = []
    pfyl_gradients = []

    for data in dataloader:
        x, c, w, z = data
        x = torch.reshape(x, (2, -1))

        for alpha in alpha_values:
            predmodel = ValueModel(alpha=alpha)
            cp = predmodel.forward(x)

            pfyl_loss = pfyl(cp, w)
            pfyl_loss.backward(retain_graph=True)
            pfyl_values.append(pfyl_loss.item())
            pfyl_gradients.append(predmodel.alpha.grad.item())

            predmodel.zero_grad()

    # Plot loss function gradients
    # Create base plot with DP solutions
    _, horizontal_plots = sp_dynamic.plot(linear=False, horizontal=True)
    # plt.grid(True)
    # plt.xlabel("Alpha")
    # plt.ylabel("Gradient")

    # Plot PFYL on top
    pfyl_grad_plot = plt.plot(alpha_values, pfyl_gradients, color="green")
    # plt.title("PFYL loss gradient vs. alpha")
    # plt.legend(
    #     [horizontal_plots, pfyl_grad_plot[0]],
    #     ["DP", "PFYL"],
    #     handler_map={tuple: HandlerTuple(ndivide=None)},
    # )

    cur_pfyl_alphas = []
    for j in range(len(alpha_values) - 1):
        if (
            pfyl_grad_plot[0].get_ydata()[j]
            <= 0
            <= pfyl_grad_plot[0].get_ydata()[j + 1]
        ):
            cur_pfyl_alphas.append(
                (
                    pfyl_grad_plot[0].get_xdata()[j]
                    + pfyl_grad_plot[0].get_xdata()[j + 1]
                )
                / 2
            )

    pfyl_alpha = np.mean(cur_pfyl_alphas)

    min_dp_value = hor_plot.get_ydata()[0]
    min_dp_range = []
    dp_alpha = None
    for hor_plot in horizontal_plots:
        cur_dp = hor_plot.get_ydata()[0]
        if cur_dp < min_dp_value:
            min_dp_value = cur_dp
            min_dp_range = hor_plot.get_xdata()
            dp_alpha = (min_dp_range[0] + min_dp_range[-1]) / 2

    epsilon = 0.2
    if pfyl_alpha is not None and dp_alpha is not None:
        tf_runs.append(
            min_dp_range[0] - epsilon < pfyl_alpha < min_dp_range[1] + epsilon
        )
        pfyl_alphas.append(pfyl_alpha)
        dp_alphas.append(dp_alpha)

        # plt.title("PFYL loss gradient vs. alpha")
        # plt.savefig("pfy_experiment.png")
        # # [hor_plot.remove() for hor_plot in horizontal_plots]
        # # pfyl_grad_plot[0].remove()
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
true_counts_pfyl, false_counts_pfyl = get_histogram_data(
    pfyl_alphas, np.array(tf_runs), bins
)
true_counts_dp, false_counts_dp = get_histogram_data(dp_alphas, np.array(tf_runs), bins)

fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Plot for pfyl_alphas
ax[0].bar(bins[:-1], true_counts_pfyl, width=width, align="edge", label="True")
ax[0].bar(
    bins[:-1],
    false_counts_pfyl,
    width=width,
    align="edge",
    bottom=true_counts_pfyl,
    label="False",
)
ax[0].set_title("pfyl_alphas")
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

plt.savefig("pfy_experiments_left.png")
