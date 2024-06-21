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


alpha_range = [-30, 30]
alpha_values = np.arange(alpha_range[0], alpha_range[1], 0.05)

small_inter = np.load("labled_data\\tiny_labled_data.npy", allow_pickle=True)

for config in small_inter:
    seed = config["seed"]
    num_items = config["num_items"]
    capacity = config["capacity"]
    lable = config["label"]
    torch.manual_seed(seed)

    weights, features, values = generate_data(
        num_items=num_items, capacity=capacity, seed=seed
    )

    optmodel = knapsackModel(weights=weights, capacity=capacity)
    data_set = dataset.optDataset(model=optmodel, feats=features, costs=values)
    dataloader = DataLoader(data_set, batch_size=1, shuffle=True)

    # Estimate gradients with dynamic programming
    features = features.reshape((2, num_items))
    dp_model = DP_Knapsack(
        weights[0], features, values, capacity, alpha_values[0], alpha_values[-1]
    )
    dp_model.solve()
    dp_model.plot(horizontal=True, linear=False)
    plt.title(lable)
    plt.show()
