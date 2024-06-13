from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np
from pyepo.data import dataset
from pyepo.func import SPOPlus, perturbedFenchelYoung
from pyepo.model.grb.knapsack import knapsackModel
import torch
from torch.utils.data import DataLoader

from data_generator import generate_data
from dp.dynamic import DP_Knapsack
from predmodel import ValueModel

from CaVEmain.src.cave import exactConeAlignedCosine
from scipy import interpolate


def fill_nan(A):
    """
    interpolate to fill nan values
    """
    A = np.array(A)
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = interpolate.interp1d(inds[good], A[good], bounds_error=False)
    B = np.where(np.isfinite(A), A, f(inds))
    return B


experiment_size = 100
num_items = 50
capacity = 30
alpha_range = [-30, 30]

alpha_values = np.arange(alpha_range[0], alpha_range[1], 0.05)
cave_sum = []
left_search = []
right_search = []
for exp in range(experiment_size):

    torch.manual_seed(exp)

    weights, features, values = generate_data(
        num_items=num_items, capacity=capacity, seed=exp
    )

    optmodel = knapsackModel(weights=weights, capacity=capacity)
    data_set = dataset.optDataset(model=optmodel, feats=features, costs=values)
    dataloader = DataLoader(data_set, batch_size=1, shuffle=True)

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
        x, c, w, z = data
        x = torch.reshape(x, (2, num_items))

        for alpha in alpha_values:
            predmodel = ValueModel(alpha=alpha)
            cp = predmodel.forward(x)

            weights_cave = torch.unsqueeze(torch.Tensor(weights), dim=0)
            cave_loss = cave(cp, weights_cave)
            cave_loss.backward(retain_graph=True)
            cave_values.append(cave_loss.item())
            cave_gradients.append(predmodel.alpha.grad.item())
        cave_gradients = fill_nan(cave_gradients)

    epsilon = 0.01
    cave_sum.append(cave_gradients)
    # check left
    for i in range(len(cave_gradients)):
        if not np.isnan(cave_gradients[i]):
            if cave_gradients[i] < epsilon:
                # going down
                if i >= 1 and (abs(cave_gradients[i - 1]) >= abs(cave_gradients[i])):
                    left_search.append(i)
                    if i <= 200:
                        print(
                            "outlier inc? i: ",
                            i,
                            "Gradient i: ",
                            cave_gradients[i],
                            "Gradient i-1: ",
                            cave_gradients[i - 1],
                        )
                        # print(cave_gradients[:250])
                        plt.plot(alpha_values[:400], cave_gradients[:400])
                        plt.savefig("outlier_cave_left_" + str(i) + "_" + str(exp))
                        plt.clf()
                    break
            else:
                left_search.append(i)
                if i <= 200:
                    print("outlier big? i: ", i, "Gradient: ", cave_gradients[i])
                break
    for i in range(len(cave_gradients) - 1, 0, -1):
        if not np.isnan(cave_gradients[i]):
            if cave_gradients[i] < epsilon:
                # going down
                if i < len(cave_gradients) - 1 and (
                    abs(cave_gradients[i]) <= abs(cave_gradients[i + 1])
                ):
                    right_search.append(i)
                    if i >= len(cave_gradients) - 200:
                        print(
                            "outlier? i: ",
                            i,
                            "Gradient i: ",
                            cave_gradients[i],
                            "Gradient i+1: ",
                            cave_gradients[i],
                        )
                        plt.plot(alpha_values[-400:], cave_gradients[-400:])
                        plt.savefig("outlier_cave_right_" + str(i) + "_" + str(exp))
                        plt.clf()
                    break
            else:
                right_search.append(i)
                if i >= len(cave_gradients) - 200:
                    print("outlier big? i: ", i, "Gradient: ", cave_gradients[i])
                break
cave_sum = np.array(cave_sum)
print(cave_sum.shape)
cave_average = cave_sum.copy()
cave_average = np.sum(cave_average, axis=0) / experiment_size
print(cave_average.shape)
variance = np.var(cave_sum, axis=0)

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(alpha_values, cave_average, color="black")
ax1.fill_between(
    alpha_values, cave_average - variance, cave_average + variance, alpha=0.5
)

dat = [x * 0.05 + alpha_range[0] for x in left_search]
ax2.hist(dat, bins=60, density=True, range=[alpha_range[0], alpha_range[1]])

dat = [x * 0.05 + alpha_range[0] for x in right_search]
ax2.hist(dat, bins=60, density=True, range=[alpha_range[0], alpha_range[1]])

plt.show()
