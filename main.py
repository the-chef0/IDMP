from matplotlib import pyplot as plt
import numpy as np
import pyepo
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

# TODO: Clean up
w1, _, v1, ni, ei = generate_data(num_items=num_items, data_points=1)
ni = torch.tensor(ni).unsqueeze(dim=0)
ei = torch.tensor(ei).unsqueeze(dim=0)
f1 = torch.cat((ni, ei), dim=1)
print(f"w1 shape {w1.shape}")
print(f"f1 shape {f1.shape}")
print(f"v1 shape {v1.shape}")
w2, f2, v2 = pyepo.data.knapsack.genData(num_data=1, num_features=200, num_items=100)
print(f"w2 shape {w2.shape}")
print(f"f2 shape {f2.shape}")
print(f"v2 shape {v2.shape}")

optmodel = knapsackModel(weights=w2, capacity=capacity)
dataset = dataset.optDataset(model=optmodel, feats=f2, costs=v2)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

spop = SPOPlus(optmodel=optmodel)
pfy = perturbedFenchelYoung(optmodel=optmodel)

alpha_values = np.arange(-5, 15, 0.05)

# Get true gradients with DP
# true_gradients = get_true_gradients(
#     weights=weights,
#     features=features,
#     values=values,
#     alpha_values=alpha_values
# )

# Estimate gradients with loss functions
spop_gradients = []
pfy_gradients = []

for data in dataloader:
    x, c, w, z = data
    x = torch.reshape(x, (2, num_items))
    print(x.shape)

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
