from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from CaVEmain.src.cave import exactConeAlignedCosine
from CaVEmain.src.dataset import optDatasetConstrs
from pyepo.data import dataset
from pyepo.data import tsp
from pyepo.model.grb.knapsack import knapsackModel
from pyepo.model.grb.tsp import tspDFJModel

from data_generator import generate_data
from predmodel import ValueModel

torch.manual_seed(100)

num_items = 100
capacity = 60

# Generate knapsack data
weights, features, values = generate_data(num_items=num_items, capacity=capacity, seed=30)
# Generate TSP data
feats, costs = tsp.genData(num_data=1, num_features=90, num_nodes=10)

optmodel_knapsack = knapsackModel(weights=weights, capacity=capacity)
optmodel_tsp = tspDFJModel(num_nodes=10)

# The knapsack dataset comes from PyEPO and does not return any binding constraints
dataset_knapsack = dataset.optDataset(model=optmodel_knapsack, feats=features, costs=values)
# The TSP dataset comes from the CaVE implementation and does return binding constraints
dataset_tsp = optDatasetConstrs(optmodel_tsp, feats, costs)
# The CaVE binding constraint dataset applied to knapsack
dataset_knapsack_bctr = optDatasetConstrs(optmodel_knapsack, features, values)

dataloader_knapsack = DataLoader(dataset_knapsack, batch_size=1, shuffle=True)
dataloader_tsp = DataLoader(dataset_tsp, batch_size=1, shuffle=True)
dataloader_knapsack_bctr = DataLoader(dataset_knapsack_bctr, batch_size=1, shuffle=True)

cave_knapsack = exactConeAlignedCosine(optmodel=optmodel_knapsack, solver="clarabel")
cave_tsp = exactConeAlignedCosine(optmodel=optmodel_tsp, solver="clarabel")

alpha_values = np.arange(-7, 7, 0.05)

cave_grad_knapsack = []
cave_grad_knapsack_bctr = []
cave_grad_tsp = []

# An example of our interpretation of how CaVE should be used
# with knapsack. We interpret the item weights as the binding constraints
# and pass those to CaVE
for data in dataloader_knapsack:
    x, c, w, z = data
    x = torch.reshape(x, (2, num_items))

    for alpha in alpha_values:
        predmodel = ValueModel(alpha=alpha)
        cp = predmodel.forward(x)

        w_cave = torch.unsqueeze(w, dim=0)
        # Binding constraints second arg
        cave_loss = cave_knapsack(cp, w_cave)
        cave_loss.backward(retain_graph=True)
        cave_grad_knapsack.append(predmodel.alpha.grad.item())

# An example of CaVE being used with TSP. This was inspired by the
# CaVE example code. Note the different dataset class (from CaVE authors), 
# and how it returns binding constraints in addition to the 4 other tensors.
for data in dataloader_tsp:
    x, _, _, _, bctr = data
    x = torch.reshape(x, (2, 45))

    for alpha in alpha_values:
        predmodel = ValueModel(alpha=alpha)
        cp = predmodel.forward(x)

        # Binding constraints second arg
        cave_loss = cave_tsp(cp, bctr)
        cave_loss.backward(retain_graph=True)
        cave_grad_tsp.append(predmodel.alpha.grad.item())

# The dataset class from the CaVE authors used with TSP can also be used
# with knapsack. Here it also returns some binding constraints, but we're
# not sure how to interpret them in the context of the knapsack problem.
# In the knapsack loop above, everything seems to work and it makes
# intuitive sense to pass the item weights as binding constraints, but
# doing it like below also works. The result looks different however,
# which makes us question what the correct way is.
for data in dataloader_knapsack_bctr:
    x, _, _, _, bctr = data
    x = torch.reshape(x, (2, num_items))

    for alpha in alpha_values:
        predmodel = ValueModel(alpha=alpha)
        cp = predmodel.forward(x)

        cave_loss = cave_knapsack(cp, bctr)
        cave_loss.backward(retain_graph=True)
        cave_grad_knapsack_bctr.append(predmodel.alpha.grad.item())

plt.plot(alpha_values, cave_grad_knapsack)
plt.plot(alpha_values, cave_grad_knapsack_bctr)
# plt.plot(alpha_values, cave_grad_tsp)
plt.savefig("binding-constraints.png")