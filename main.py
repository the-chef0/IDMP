from matplotlib import pyplot as plt
import numpy as np
from pyepo.data import dataset
from pyepo.func import SPOPlus, perturbedFenchelYoung
from pyepo.model.grb.knapsack import knapsackModel
import torch
from torch.utils.data import DataLoader

from data_generator import generate_data
from dynamic_programming import get_true_gradients
from predmodel import ValueModel

num_items = 100
capacity = 30

# TODO: Adapt to Miquel's data generation, but that might not be necessary
# because his function will have to return the same shapes as PyEPO for
# the optDataset instance to be correctly created
weights, features, values = generate_data(num_items=num_items)
optmodel = knapsackModel(weights=weights, capacity=capacity)
dataset = dataset.optDataset(model=optmodel, feats=features, costs=values)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

spop = SPOPlus(optmodel=optmodel)
pfy = perturbedFenchelYoung(optmodel=optmodel)

alpha_values = np.arange(-5, 15, 0.05)

# Get true gradients with DP
true_gradients = get_true_gradients(
    weights=weights,
    features=features,
    values=values,
    alpha_values=alpha_values
)

# Estimate gradients with loss functions
spop_gradients = []
pfy_gradients = []

for data in dataloader:
    x, c, w, z = data
    x = torch.reshape(x, (2, num_items))

    for alpha in alpha_values:
        predmodel = ValueModel(alpha=alpha)
        cp = predmodel.forward(x)

        spop_loss = spop(cp, c, w, z)
        spop_loss.backward(retain_graph=True)
        spop_gradients.append(predmodel.alpha.grad.item())

        pfy_loss = pfy(cp, w)
        pfy_loss.backward(retain_graph=True)
        pfy_gradients.append(predmodel.alpha.grad.item())

plt.plot(alpha_values, spop_gradients)
plt.plot(alpha_values, pfy_gradients)
plt.savefig("grads.png")
