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

experiment_size = 50
num_items = 50
capacity = 30

for i in range(experiment_size):

    torch.manual_seed(i)

    weights, features, values = generate_data(num_items=num_items, capacity=capacity, seed=i)

    optmodel = knapsackModel(weights=weights, capacity=capacity)
    dataset = dataset.optDataset(model=optmodel, feats=features, costs=values)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    cave = exactConeAlignedCosine(optmodel=optmodel, solver="clarabel")

    alpha_values = np.arange(-7, 7, 0.05)

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

            w_cave = torch.unsqueeze(w, dim=0)
            cave_loss = cave(cp, w_cave)
            cave_loss.backward(retain_graph=True)
            cave_values.append(cave_loss.item())
            cave_gradients.append(predmodel.alpha.grad.item())

