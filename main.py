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
from CaVEmain.src.dataset import optDatasetConstrs, collate_fn
torch.manual_seed(100)

num_items = 50
capacity = 30

weights, features, values = generate_data(num_items=num_items, capacity=capacity, seed=37)

optmodel = knapsackModel(weights=weights, capacity=capacity)
dataset = optDatasetConstrs(model=optmodel, feats=features, costs=values)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)

spop = SPOPlus(optmodel=optmodel)
pfy = perturbedFenchelYoung(optmodel=optmodel)
cave = exactConeAlignedCosine(optmodel=optmodel, solver="clarabel")

alpha_values = np.arange(-7, 7, 0.05)

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
cave_values_ctr = []
cave_gradients = []
cave_gradients_ctr = []

for data in dataloader:
    x, c, w, z, bctr = data
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
        cave_loss_ctr = cave(cp, bctr)
        cave_loss_ctr.backward(retain_graph=True)
        cave_values_ctr.append(cave_loss_ctr.item() * 100)
        cave_gradients_ctr.append(predmodel.alpha.grad.item() * 100)

        predmodel.zero_grad()
        cave_loss = cave(cp, torch.unsqueeze(torch.tensor(weights), dim=0))
        cave_loss.backward(retain_graph=True)
        cave_values.append(cave_loss.item() * 100)
        cave_gradients.append(predmodel.alpha.grad.item() * 100)

# Plot loss function gradients
# Create base plot with DP solutions
linear_plots, horzizontal_plots, loss_plots, intervals = dp_model.plot(linear=True, horizontal=True, loss=True, z=torch.squeeze(z, dim=0).item())
plt.grid(True)
plt.xlabel("Alpha")
plt.ylabel("loss")
# Plot SPO loss
spop_loss_plot = plt.plot(alpha_values, spop_values, color="green", label='SPO+')
cave_loss_plot = plt.plot(alpha_values, cave_values, color='magenta', label='CaVE')
cave_loss_plot = plt.plot(alpha_values, cave_values_ctr, color='purple', label='CaVE_ctr')
pfyl_loss_plot = plt.plot(alpha_values, pfy_values, color='blue', label='PFYL')
for interval, loss in zip(intervals, loss_plots):
    plt.plot(interval, loss, '--', color='red',)
plt.legend(['SPO+', 'CaVE', 'CaVE_ctr' 'pfyl', 'SPO'])
plt.savefig("true_spo_loss.png")

plt.clf()

spop_loss_plot = plt.plot(alpha_values, spop_gradients, color="green", label='SPO+')
cave_loss_plot = plt.plot(alpha_values, cave_gradients, color='magenta', label='CaVE')
cave_loss_plot = plt.plot(alpha_values, cave_gradients_ctr, color='purple', label='CaVE_ctr')
pfyl_loss_plot = plt.plot(alpha_values, pfy_gradients, color='blue', label='PFYL')
for interval, loss in zip(intervals, horzizontal_plots):
    plt.plot(interval, loss, '--', color='red',)
plt.legend(['SPO+', 'CaVE', 'CaVE_ctr', 'pfyl', 'DP'])
plt.savefig("gradients.png")
# Remove SPO+ to create the next plot
# spop_loss_plot[0].remove()
# Plot SPO on top
# spop_grad_plot = plt.plot(alpha_values, spop_gradients, color="green")
# plt.title("SPO+ loss gradient vs. alpha")
# plt.legend(
#     [horizontal_plots, spop_grad_plot[0]],
#     ["DP", "SPO+"],
#     handler_map={tuple: HandlerTuple(ndivide=None)},
# )
# plt.savefig("spo_grad.png")
# # Remove SPO+ to create the next plot
# spop_grad_plot[0].remove()

# pfy_grad_plot = plt.plot(alpha_values, pfy_gradients, color="blue")
# plt.title("PFYL gradient vs. alpha")
# plt.legend(
#     [horizontal_plots, pfy_grad_plot[0]],
#     ["DP", "PFYL"],
#     handler_map={tuple: HandlerTuple(ndivide=None)},
# )
# plt.savefig("pfy_grad.png")
# pfy_grad_plot[0].remove()


# cave_grad_plot = plt.plot(alpha_values, spop_gradients, color="green")
# plt.title("CaVE loss gradient vs. alpha")
# plt.legend(
#     [horizontal_plots, pfy_grad_plot[0]],
#     ["DP", "SPO+"],
#     handler_map={tuple: HandlerTuple(ndivide=None)},
# )
# plt.savefig("SPO_grad.png")
# cave_grad_plot[0].remove()
