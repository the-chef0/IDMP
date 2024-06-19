from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np
from pyepo.data import dataset
from pyepo.data.tsp import genData
from pyepo.model.grb.knapsack import knapsackModel
import torch
from torch.utils.data import DataLoader

from data_generator import generate_data
from dp.dynamic import DP_Knapsack
from predmodel import ValueModel
from scipy import interpolate
from CaVEmain.src.cave import exactConeAlignedCosine
from CaVEmain.src.dataset import optDatasetConstrs, collate_fn
import sys
from CaVEmain.src.model.tsp import tspDFJModel
# sys.path.append("CaVEmain/src/model")
# from tsp import tspDFJModel
import pickle

def generate_tsp_data(seed, num_node):
    num_data = 1 # number of training data
    num_feat = 2 * (num_node * (num_node - 1) // 2)# size of feature
    poly_deg = 1 # polynomial degree
    noise = 0.5 # noise width
    feats, costs = genData(num_data, num_feat, num_node, poly_deg, noise, seed=seed)
    print(feats.shape, costs.shape)
    return feats, costs




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
#ranges = [2,1,0.8,0.7]
ranges = [4]
#how many problems
experiment_size = 10

#knapsack params
num_items = 10
capacity = 20

# tsp params
num_node = 5 # node size

res = np.empty((len(degrees), len(ranges)))
res_tsp = np.empty((len(degrees), len(ranges)))
for d_ind, deg in enumerate(degrees):
    l_infs = []
    l_infs_relative = []
    MSE_vals = []
    for r_ind, r in enumerate(ranges):
        cave_gradients_total = []
        cave_values_total = []
        cubic_fits = []

        cave_tsp_gradients_total = []
        cave_tsp_values_total = []
        cubic_tsp_fits = []

        alpha_values = np.arange(-r, r, 0.05)
        for i in range(experiment_size):

            torch.manual_seed(i)
            
            weights, features, values = generate_data(num_items=num_items, capacity=capacity, seed=i)
            x_tsp, c_tsp = generate_tsp_data(seed=i, num_node=num_node)

            optmodel = knapsackModel(weights=weights, capacity=capacity)
            data_set = optDatasetConstrs(model=optmodel, feats=features, costs=values)
            dataloader = DataLoader(data_set, batch_size=1, shuffle=True)

            optmodel_tsp = tspDFJModel(num_node)
            dataset_tsp = optDatasetConstrs(optmodel_tsp, x_tsp, c_tsp)
            dataloader_tsp = DataLoader(dataset_tsp, batch_size=1, collate_fn=collate_fn, shuffle=True)

            cave = exactConeAlignedCosine(optmodel=optmodel, solver="clarabel")
            cave_tsp = exactConeAlignedCosine(optmodel=optmodel_tsp, solver="clarabel")

            cave_values = []
            cave_gradients = []

            cave_tsp_values = []
            cave_tsp_gradients = []

            for data in dataloader:
                x, c, w, z, bctr = data
                x = torch.reshape(x, (2, num_items))

                for alpha in alpha_values:
                    predmodel = ValueModel(alpha=alpha)
                    cp = predmodel.forward(x)

                    cave_loss = cave(cp, bctr)
                    cave_loss.backward(retain_graph=True)
                    cave_values.append(cave_loss.item())
                    cave_gradients.append(predmodel.alpha.grad.item())

            for data in dataloader_tsp:
                x, _, _, _, bctr = data
                x = torch.reshape(x, (2, -1))

                for alpha in alpha_values:
                    predmodel = ValueModel(alpha=alpha)
                    cp = predmodel.forward(x)

                    cave_loss = cave_tsp(cp, bctr)
                    cave_loss.backward(retain_graph=True)
                    cave_tsp_values.append(cave_loss.item())
                    cave_tsp_gradients.append(predmodel.alpha.grad.item())

            safe_gradients = fill_nan(cave_gradients)
            cave_gradients_total.append(safe_gradients)
            cave_values_total.append(cave_values)
            fit = np.polynomial.polynomial.Polynomial.fit(alpha_values, safe_gradients, deg)
            cubic_fits.append(fit)

            safe_tsp_gradients = fill_nan(cave_tsp_gradients)
            cave_tsp_gradients_total.append(safe_tsp_gradients)
            cave_tsp_values_total.append(cave_tsp_values)
            fit = np.polynomial.polynomial.Polynomial.fit(alpha_values, safe_tsp_gradients, deg)
            cubic_tsp_fits.append(fit)

        # #create plots
        for i in range(experiment_size):
            plt.plot(alpha_values, cave_gradients_total[i], label='CaVE')
            xx, yy = cubic_fits[i].linspace()
            plt.plot(xx, yy, label='fit')
            # plt.plot(alpha_values, cave_values_total[i])
            plt.savefig(f'cave_fit_{i}')
            plt.clf()

        for i in range(experiment_size):
            plt.plot(alpha_values, cave_tsp_gradients_total[i], label='CaVE')
            xx, yy = cubic_tsp_fits[i].linspace()
            plt.plot(xx, yy, label='fit')
            # plt.plot(alpha_values, cave_values_total[i])
            plt.savefig(f'cave_tsp_fit_{i}')
            plt.clf()


        #calc relative l_inf norms
        # for i in range(experiment_size):
        #     fit = cubic_fits[i]
        #     grads = cave_gradients_total[i]
        #     errors = []
        #     max_dist = np.max(grads) - np.min(grads)
        #     for idx, a_val in enumerate(alpha_values):
        #         errors.append(np.abs(grads[idx] - fit(a_val)))
        #     l_infs.append(np.max(errors))
        #     l_infs_relative.append(np.max(errors)/max_dist * 100)
        #     MSE_vals.append(np.mean(np.power(errors, 2)))

        # res[d_ind,r_ind] = np.mean(l_infs)

        # for i in range(experiment_size):
        #     fit = cubic_fits[i]
        #     grads = cave_gradients_total[i]
        #     errors = []
        #     max_dist = np.max(grads) - np.min(grads)
        #     for idx, a_val in enumerate(alpha_values):
        #         errors.append(np.abs(grads[idx] - fit(a_val)))
        #     l_infs.append(np.max(errors))
        #     l_infs_relative.append(np.max(errors)/max_dist * 100)
        #     MSE_vals.append(np.mean(np.power(errors, 2)))

        # res[d_ind,r_ind] = np.mean(l_infs)

        # print(f'l_infinity: mean: {np.mean(l_infs)}, median: {np.median(l_infs)}, max: {np.max(l_infs)}, variance: {np.var(l_infs)}')
        # print(f'relative l_infinity: mean: {np.mean(l_infs_relative)}, median: {np.median(l_infs_relative)}, max: {np.max(l_infs_relative)}, variance: {np.var(l_infs_relative)}')
        # print(f'MSE: mean: {np.mean(MSE_vals)}, median: {np.median(MSE_vals)}, max: {np.max(MSE_vals)}, variance: {np.var(MSE_vals)}')

        # with open('l_infs.pickle', 'wb') as handle:
        #     pickle.dump(l_infs, handle)
        # with open('l_infs_relative.pickle', 'wb') as handle:
        #     pickle.dump(l_infs_relative, handle)
        # with open('mse_vals.pickle', 'wb') as handle:
        #     pickle.dump(MSE_vals, handle)

# with open('art_graph_data.pickle', 'wb') as handle:
#     pickle.dump(res, handle)

# for i in range(res.shape[1]):
#     plt.plot(degrees, res[:, i].reshape(-1), label=f'<{-ranges[i]}, {ranges[i]}>')
# plt.xlabel('Degree of polynomial fit')
# plt.ylabel('L infinity norm')
# plt.legend()
# plt.savefig('art_graph')