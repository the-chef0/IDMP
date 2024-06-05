from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np
from pyepo.data import dataset
from pyepo.model.grb.knapsack import knapsackModel
import torch
from torch.utils.data import DataLoader

from data_generator import generate_data
from dp.dynamic import DP_Knapsack
from predmodel import ValueModel
from scipy import interpolate
from CaVEmain.src.cave import exactConeAlignedCosine

def fill_nan(A):
    """
    interpolate to fill nan values
    """
    A = np.array(A)
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = interpolate.interp1d(inds[good], A[good], bounds_error=False)
    B = np.where(np.isfinite(A), A, f(inds))
    return B

degrees = [0,1,2,3,4,5]
#degrees = [3]
experiment_size = 10
num_items = 50
capacity = 30
average_MSEs = []
for deg in degrees:

    cave_gradients_total = []
    cubic_fits = []
    alpha_values = np.arange(-1, 1, 0.05)
    for i in range(experiment_size):

        torch.manual_seed(i)

        weights, features, values = generate_data(num_items=num_items, capacity=capacity, seed=i)

        optmodel = knapsackModel(weights=weights, capacity=capacity)
        data_set = dataset.optDataset(model=optmodel, feats=features, costs=values)
        dataloader = DataLoader(data_set, batch_size=1, shuffle=True)

        cave = exactConeAlignedCosine(optmodel=optmodel, solver="clarabel")

        # Estimate gradients with loss functions

        cave_values = []
        cave_gradients = []

        for data in dataloader:
            x, c, w, z = data
            x = torch.reshape(x, (2, num_items))

            for alpha in alpha_values:
                predmodel = ValueModel(alpha=alpha)
                cp = predmodel.forward(x)

                weights_cave = torch.unsqueeze(weights, dim=0)
                cave_loss = cave(cp, weights_cave)
                cave_loss.backward(retain_graph=True)
                cave_values.append(cave_loss.item())
                cave_gradients.append(predmodel.alpha.grad.item())

        safe_gradients = fill_nan(cave_gradients)
        scaled_safe_gradients = safe_gradients * 100
        cave_gradients_total.append(scaled_safe_gradients)
        fit = np.polynomial.polynomial.Polynomial.fit(alpha_values, scaled_safe_gradients, deg)
        cubic_fits.append(fit)

    #create plots
    for i in range(experiment_size):
        plt.plot(alpha_values, cave_gradients_total[i])
        xx, yy = cubic_fits[i].linspace()
        plt.plot(xx, yy)
        plt.savefig(f'cave_fit_{i}_1')
        plt.clf()

    #calc MSE vals
    MSE_vals = []
    for i in range(experiment_size):
        fit = cubic_fits[i]
        grads = cave_gradients_total[i]
        error_sum = 0
        for idx, a_val in enumerate(alpha_values):
            error_sum += (grads[idx] - fit(a_val))**2
        MSE_vals.append(error_sum/len(alpha_values))

    average_MSEs.append(np.mean(MSE_vals))

print(average_MSEs)