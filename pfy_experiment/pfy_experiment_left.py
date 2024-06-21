import torch
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from pyepo.data import dataset
from pyepo.func import perturbedFenchelYoung
from pyepo.model.grb.knapsack import knapsackModel
from torch.utils.data import DataLoader
from data_generator import generate_data
from dp.dynamic import DP_Knapsack
from predmodel import ValueModel

left_data = np.load("../labled_data/left_labled_data.npy", allow_pickle=True)
num_runs = len(left_data)
tf_runs = []
pfyl_alphas = []
dp_alphas = []
alpha_values = np.arange(left_data[0]['alpha'][0], left_data[0]['alpha'][1], 0.05)
num_items = left_data[0]['num_items']
capacity = left_data[0]['capacity']
for i in range(num_runs):
    if i % 10 == 0:
        print(f"Running iteration: {i}")
    torch.manual_seed(left_data[i]['seed'])

    weights, features, values = generate_data(num_items=num_items, capacity=capacity, seed=left_data[i]['seed'])

    optmodel = knapsackModel(weights=weights, capacity=capacity)
    data = dataset.optDataset(model=optmodel, feats=features, costs=values)
    dataloader = DataLoader(data, batch_size=1, shuffle=True)

    pfyl = perturbedFenchelYoung(optmodel=optmodel)

    # Estimate gradients with dynamic programming
    features = features.reshape((2, num_items))
    dp_model = DP_Knapsack(
        weights[0], features, values, capacity, alpha_values[0], alpha_values[-1]
    )
    dp_model.solve()

    # Estimate gradients with loss functions
    pfyl_values = []
    pfyl_gradients = []

    for data in dataloader:
        x, c, w, z = data
        x = torch.reshape(x, (2, num_items))

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
    _, horizontal_plots = dp_model.plot(linear=False, horizontal=True)
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
        if pfyl_grad_plot[0].get_ydata()[j] <= 0 <= pfyl_grad_plot[0].get_ydata()[j + 1]:
            cur_pfyl_alphas.append((pfyl_grad_plot[0].get_xdata()[j] + pfyl_grad_plot[0].get_xdata()[j + 1]) / 2)

    pfyl_alpha = np.mean(cur_pfyl_alphas)

    max_dp_value = 0
    max_dp_range = []
    dp_alpha = None
    for hor_plot in horizontal_plots:
        cur_dp = hor_plot.get_ydata()[0]
        if cur_dp > max_dp_value:
            max_dp_value = cur_dp
            max_dp_range = hor_plot.get_xdata()
            dp_alpha = (max_dp_range[0] + max_dp_range[-1]) / 2

    epsilon = 0.2
    if pfyl_alpha is not None and dp_alpha is not None:
        tf_runs.append(max_dp_range[0] - epsilon < pfyl_alpha < max_dp_range[1] + epsilon)
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
bins = np.arange(left_data[0]['alpha'][0], left_data[0]['alpha'][1], width)
true_counts_pfyl, false_counts_pfyl = get_histogram_data(pfyl_alphas, np.array(tf_runs), bins)
true_counts_dp, false_counts_dp = get_histogram_data(dp_alphas, np.array(tf_runs), bins)

fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Plot for pfyl_alphas
ax[0].bar(bins[:-1], true_counts_pfyl, width=width, align='edge', label='True')
ax[0].bar(bins[:-1], false_counts_pfyl, width=width, align='edge', bottom=true_counts_pfyl, label='False')
ax[0].set_title('pfyl_alphas')
ax[0].set_ylabel('Count')
ax[0].legend()

# Plot for dp_alphas
ax[1].bar(bins[:-1], true_counts_dp, width=width, align='edge', label='True')
ax[1].bar(bins[:-1], false_counts_dp, width=width, align='edge', bottom=true_counts_dp, label='False')
ax[1].set_title('dp_alphas')
ax[1].set_xlabel('Alpha')
ax[1].set_ylabel('Count')
ax[1].legend()

plt.savefig('pfy_experiments_left.png')
