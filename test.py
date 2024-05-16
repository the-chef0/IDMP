from pyepo.data import knapsack, dataset
from pyepo.func import SPOPlus
from pyepo.model.grb.knapsack import knapsackModel
from torch.utils.data import DataLoader
import torch

from predmodel import ValueModel

weights, feats, costs = knapsack.genData(num_data=1, num_features=10, num_items=5)

optmodel = knapsackModel(weights=weights, capacity=12)
spo = SPOPlus(optmodel, processes=2)

dataset = dataset.optDataset(model=optmodel, feats=feats, costs=costs)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

alpha_options = [*range(-50, 50, 1)]

losses = []

for data in dataloader:
    x, c, w, z = data
    x = torch.reshape(x, (2, 5))

    for alpha in alpha_options:
        predmodel = ValueModel(alpha=alpha)
        cp = predmodel.forward(x)
        loss = spo(cp, c, w, z)
        losses.append(loss.detach())
        loss.backward()
        # print(predmodel.alpha.grad.view(-1))

print(losses)