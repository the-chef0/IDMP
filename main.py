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


num_items = 100
capacity = 30

weights, features, values = generate_data(num_items=num_items, capacity=capacity)

optmodel = knapsackModel(weights=weights, capacity=capacity)
dataset = dataset.optDataset(model=optmodel, feats=features, costs=values)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

spop = SPOPlus(optmodel=optmodel)
pfy = perturbedFenchelYoung(optmodel=optmodel)
cave = exactConeAlignedCosine(optmodel=optmodel, solver="clarabel")

alpha_values = np.arange(-15, 15, 0.05)

# Estimate gradients with dynamic programming
features = features.reshape((2, num_items))
dp_model = DP_Knapsack(
    weights[0], features, values, capacity, alpha_values[0], alpha_values[-1]
)
dp_model.solve()

# Estimate gradients with loss functions
spop_values = []
spop_gradients = []
pfy_values = []
pfy_gradients = []
cave_values = []
cave_gradients = []

for data in dataloader:
    x, c, w, z = data
    x = torch.reshape(x, (2, num_items))

    for alpha in alpha_values:
        predmodel = ValueModel(alpha=alpha)
        cp = predmodel.forward(x)

        spop_loss = spop(cp, c, w, z)
        spop_loss.backward(retain_graph=True)
        spop_values.append(spop_loss.item())
        spop_gradients.append(predmodel.alpha.grad.item())

        predmodel.zero_grad()

        pfy_loss = pfy(cp, w)
        pfy_loss.backward(retain_graph=True)
        pfy_values.append(pfy_loss.item())
        pfy_gradients.append(predmodel.alpha.grad.item())

        predmodel.zero_grad()

        w_cave = torch.unsqueeze(w, dim=0)
        cave_loss = cave(cp, w_cave)
        cave_loss.backward(retain_graph=True)
        cave_values.append(cave_loss.item())
        cave_gradients.append(predmodel.alpha.grad.item())

# Plot loss function gradients
_, horizontal_plots = dp_model.plot(linear=False, horizontal=True)
spop_grad_plot = plt.plot(alpha_values, spop_gradients, color="green")
pfy_grad_plot = plt.plot(alpha_values, pfy_gradients, color="blue")
cave_grad_plot = plt.plot(alpha_values, cave_gradients, color="magenta")
plt.grid(True)
plt.title("Loss function gradients vs. alpha")
plt.xlabel("Alpha")
plt.ylabel("Gradient")
plt.legend(
    [horizontal_plots, spop_grad_plot[0], pfy_grad_plot[0], cave_grad_plot[0]],
    ["DP", "SPO+", "PFYL", "CaVE"],
    handler_map={tuple: HandlerTuple(ndivide=None)},
)
plt.savefig("gradients.png")
plt.clf()

# Plot loss function values
linear_plots, _ = dp_model.plot(linear=True, horizontal=False)
spop_value_plot = plt.plot(alpha_values, spop_values, color="green")
pfy_value_plot = plt.plot(alpha_values, pfy_values, color="blue")
cave_value_plot = plt.plot(alpha_values, cave_values, color="magenta")
plt.grid(True)
plt.title("Loss function values vs. alpha")
plt.xlabel("Alpha")
plt.ylabel("Loss value")
plt.legend(
    [linear_plots, spop_value_plot[0], pfy_value_plot[0], cave_value_plot[0]],
    ["DP", "SPO+", "PFYL", "CaVE"],
    handler_map={tuple: HandlerTuple(ndivide=None)},
)
plt.savefig("values.png")
plt.clf()


cave_grad_plot = plt.plot(alpha_values, cave_gradients, color="magenta")
plt.grid(True)
plt.title("Loss function gradients vs. alpha")
plt.xlabel("Alpha")
plt.ylabel("Gradient")
plt.savefig("gradients_cave.png")
plt.clf()

# Plot loss function values
cave_value_plot = plt.plot(alpha_values, cave_values, color="magenta")
plt.grid(True)
plt.title("Loss function values vs. alpha")
plt.xlabel("Alpha")
plt.ylabel("Loss value")
plt.savefig("values_cave.png")
plt.clf()
