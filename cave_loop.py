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
import pickle

def fill_nan(A):
    """
    interpolate to fill nan values
    """
    A = np.array(A)
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = interpolate.interp1d(inds[good], A[good], bounds_error=False, fill_value='extrapolate')
    B = np.where(np.isfinite(A), A, f(inds))
    return B

#degrees = [0,1,2,3,4,5]
degrees = [3]
experiment_size = 10
num_items = 50
capacity = 30
l_infs = []
l_infs_relative = []
MSE_vals = []
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

                weights_cave = torch.unsqueeze(torch.tensor(weights), dim=0)
                cave_loss = cave(cp, weights_cave)
                cave_loss.backward(retain_graph=True)
                cave_values.append(cave_loss.item())
                cave_gradients.append(predmodel.alpha.grad.item())

        safe_gradients = fill_nan(cave_gradients)
        scaled_safe_gradients = safe_gradients * 100
        cave_gradients_total.append(scaled_safe_gradients)
        fit = np.polynomial.polynomial.Polynomial.fit(alpha_values, scaled_safe_gradients, deg)
        cubic_fits.append(fit)

    # #create plots
    # for i in range(experiment_size):
    #     plt.plot(alpha_values, cave_gradients_total[i])
    #     xx, yy = cubic_fits[i].linspace()
    #     plt.plot(xx, yy)
    #     plt.savefig(f'cave_fit_{i}_1')
    #     plt.clf()

    #calc relative l_inf norms
    for i in range(experiment_size):
        fit = cubic_fits[i]
        grads = cave_gradients_total[i]
        errors = []
        range = np.max(grads) - np.min(grads)
        for idx, a_val in enumerate(alpha_values):
            errors.append(np.abs(grads[idx] - fit(a_val)))
        l_infs.append(np.max(errors))
        l_infs_relative.append(np.max(errors)/range * 100)
        MSE_vals.append(np.mean(errors **2))
    print(f'l_infinity: mean: {np.mean(l_infs)}, median: {np.median(l_infs)}, max: {np.max(l_infs)}, variance: {np.var(l_infs)}')
    print(f'relative l_infinity: mean: {np.mean(l_infs_relative)}, median: {np.median(l_infs_relative)}, max: {np.max(l_infs_relative)}, variance: {np.var(l_infs_relative)}')
    print(f'MSE: mean: {np.mean(MSE_vals)}, median: {np.median(MSE_vals)}, max: {np.max(MSE_vals)}, variance: {np.var(MSE_vals)}')

    with open('l_infs.pickle', 'wb') as handle:
        pickle.dump(l_infs, handle)
    with open('l_infs_relative.pickle', 'wb') as handle:
        pickle.dump(l_infs_relative, handle)
    with open('mse_vals.pickle', 'wb') as handle:
        pickle.dump(MSE_vals, handle)