import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from pyepo.data import dataset
from pyepo.func import perturbedFenchelYoung, SPOPlus
from pyepo.model.grb.knapsack import knapsackModel
from torch.utils.data import DataLoader
from data_generator import generate_data
from dp.dynamic import DP_Knapsack
from predmodel import ValueModel
from operator import itemgetter


def find_jumps(gradients, alpha_values, epsilon):
    jumps = []
    for i in range(1, len(gradients)):
        jump = abs(gradients[i] - gradients[i - 1])
        if jump > epsilon:
            jumps.append((alpha_values[i], jump))
    return jumps


def get_largest_jumps(jumps, n):
    jumps.sort(key=lambda x: abs(x[1]), reverse=True)
    return jumps[:n]


def compare_jumps(spop_jumps, pfyl_jumps, range_percent):
    if len(spop_jumps) != len(pfyl_jumps):
        return False
    spo_alpha_norm = max(spop_jumps, key=itemgetter(1))[0] - min(spop_jumps, key=itemgetter(1))[0]
    spo_jump_norm = max(spop_jumps, key=itemgetter(1))[1] - min(spop_jumps, key=itemgetter(1))[1]
    pfy_alpha_norm = max(pfyl_jumps, key=itemgetter(1))[0] - min(pfyl_jumps, key=itemgetter(1))[0]
    pfy_jump_norm = max(pfyl_jumps, key=itemgetter(1))[1] - min(pfyl_jumps, key=itemgetter(1))[1]
    for (alpha1, jump1), (alpha2, jump2) in zip(spop_jumps, pfyl_jumps):
        # print(((alpha1 / spo_alpha_norm) / (jump1 / spo_jump_norm)) / (
        #         (alpha2 / pfy_alpha_norm) / (jump2 / pfy_jump_norm)))
        if abs(((alpha1 / spo_alpha_norm) / (jump1 / spo_jump_norm)) / (
                (alpha2 / pfy_alpha_norm) / (jump2 / pfy_jump_norm))) > range_percent / 100:
            return False
    return True


middle_data = np.load("labled_data/middle_labled_data.npy", allow_pickle=True)
num_runs = len(middle_data)
total_correct = 0
pfyl_alphas = []
dp_alphas = []
alpha_values = np.arange(middle_data[0]['alpha'][0], middle_data[0]['alpha'][1], 0.05)
num_items = middle_data[0]['num_items']
capacity = middle_data[0]['capacity']
for i in range(num_runs):
    # if i % 10 == 0:
    #     print(f"Running iteration: {i}")
    torch.manual_seed(middle_data[i]['seed'])

    weights, features, values = generate_data(num_items=num_items, capacity=capacity, seed=middle_data[i]['seed'])

    optmodel = knapsackModel(weights=weights, capacity=capacity)
    data = dataset.optDataset(model=optmodel, feats=features, costs=values)
    dataloader = DataLoader(data, batch_size=1, shuffle=True)

    spop = SPOPlus(optmodel=optmodel)
    pfyl = perturbedFenchelYoung(optmodel=optmodel, sigma=0.01)

    # Estimate gradients with dynamic programming
    features = features.reshape((2, num_items))
    dp_model = DP_Knapsack(
        weights[0], features, values, capacity, alpha_values[0], alpha_values[-1]
    )
    dp_model.solve()

    # Estimate gradients with loss functions
    spop_gradients = []
    pfyl_gradients = []

    for data in dataloader:
        x, c, w, z = data
        x = torch.reshape(x, (2, num_items))

        for alpha in alpha_values:
            predmodel = ValueModel(alpha=alpha)
            cp = predmodel.forward(x)

            pfyl_loss = pfyl(cp, w)
            pfyl_loss.backward(retain_graph=True)
            pfyl_gradients.append(predmodel.alpha.grad.item())

            predmodel.zero_grad()

            spop_loss = spop(cp, c, w, z)
            spop_loss.backward(retain_graph=True)
            spop_gradients.append(predmodel.alpha.grad.item())

            predmodel.zero_grad()

    epsilon = 0.1
    num_jumps = 5
    range_percent = 10

    spop_jumps = find_jumps(spop_gradients, alpha_values, epsilon)
    pfyl_jumps = find_jumps(pfyl_gradients, alpha_values, epsilon)

    largest_spop_jumps = get_largest_jumps(spop_jumps, num_jumps)
    largest_pfyl_jumps = get_largest_jumps(pfyl_jumps, num_jumps)

    result = compare_jumps(largest_spop_jumps, largest_pfyl_jumps, range_percent)
    if result:
        total_correct += 1
    print(f"similar structure on run {i} / {num_runs} is {result} with total correct: {total_correct}")

    # Plot loss function gradients
    # Create base plot with DP solutions
    _, horizontal_plots = dp_model.plot(linear=False, horizontal=True)
    plt.grid(True)
    plt.xlabel("Alpha")
    plt.ylabel("Gradient")

    # Plot SPO+ on top
    spop_grad_plot = plt.plot(alpha_values, spop_gradients, color="blue")
    pfyl_grad_plot = plt.plot(alpha_values, pfyl_gradients, color="green")
    plt.title("PFYL loss gradient vs. alpha")
    plt.legend(
        [horizontal_plots, spop_grad_plot[0], pfyl_grad_plot[0]],
        ["DP", "SPO+", "PFYL"],
        handler_map={tuple: HandlerTuple(ndivide=None)},
    )

    plt.title("SPO+ loss gradient vs. alpha")
    if result:
        plt.savefig("pfy_plots/pfy_experiment" + str(i) + ".png")
    else:
        plt.savefig("pfy_experiment.png")
    plt.clf()
