from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np
from pyepo.data import dataset
from pyepo.func import SPOPlus, perturbedFenchelYoung
from pyepo.model.grb.knapsack import knapsackModel
import torch
from torch.utils.data import DataLoader
from pyepo.model.grb.shortestpath import shortestPathModel
import pyepo.data.shortestpath
from data_generator import generate_data
from dp.SP_dynamic import SP_dynamic
from dp.dynamic import DP_Knapsack
from predmodel import ValueModel


alpha_range = [-30, 30]
alpha_values = np.arange(alpha_range[0], alpha_range[1], 0.05)

small_inter = np.load("labled_data\\SP_left_labled_data.npy", allow_pickle=True)

# SP load
for config in small_inter:
    num_data = 1
    seed = config["seed"]
    grid = config["grid_size"]
    num_feat = 2 * ((grid[0] - 1) * grid[1] + (grid[1] - 1) * grid[0])
    lable = config["label"]
    torch.manual_seed(seed)

    x, c = pyepo.data.shortestpath.genData(
        num_data, num_feat, grid, deg=1, noise_width=0, seed=seed
    )
    optmodel = shortestPathModel(grid=grid)
    data = dataset.optDataset(model=optmodel, feats=x, costs=c)
    dataloader = DataLoader(data, batch_size=1, shuffle=True)

    x = x.reshape((2, -1))
    sp_dynamic = SP_dynamic(x, c, grid, -5, 5)
    sp_dynamic.solve()

    linear_plots, horizontal_plots, loss_plots, intervals = sp_dynamic.plot(
        horizontal=True, linear=False
    )
    for interval, loss in zip(intervals, horizontal_plots):
        plt.plot(
            interval,
            loss,
            "--",
            color="red",
        )
    plt.title(lable)
    plt.show()
# Knapsack load
# for config in small_inter:
#     seed = config["seed"]
#     num_items = config["num_items"]
#     capacity = config["capacity"]
#     lable = config["label"]
#     torch.manual_seed(seed)

#     weights, features, values = generate_data(
#         num_items=num_items, capacity=capacity, seed=seed
#     )

#     optmodel = knapsackModel(weights=weights, capacity=capacity)
#     data_set = dataset.optDataset(model=optmodel, feats=features, costs=values)
#     dataloader = DataLoader(data_set, batch_size=1, shuffle=True)

#     # Estimate gradients with dynamic programming
#     features = features.reshape((2, num_items))
#     dp_model = DP_Knapsack(
#         weights[0], features, values, capacity, alpha_values[0], alpha_values[-1]
#     )
#     dp_model.solve()
#     dp_model.plot(horizontal=True, linear=False)
#     plt.title(lable)
#     plt.show()
