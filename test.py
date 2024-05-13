from pyepo.data import knapsack, dataset
from pyepo.func import SPOPlus
from pyepo.model.grb.knapsack import knapsackModel
from torch.utils.data import DataLoader

from predmodel import ValueModel

weights, feats, costs = knapsack.genData(num_data=5, num_features=2, num_items=5)

optmodel = knapsackModel(weights=weights, capacity=12)
spo = SPOPlus(optmodel, processes=2)

dataset = dataset.optDataset(model=optmodel, feats=feats, costs=costs)
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

alpha_options = [0.2, 0.4, 0.6]

for alpha in alpha_options:
    predmodel = ValueModel(alpha=alpha)

    for data in dataloader:
        x, c, w, z = data
        cp = predmodel.forward(x)
        loss = spo(cp, c, w, z)
        loss.backward()
        print(predmodel.alpha.grad.view(-1))
