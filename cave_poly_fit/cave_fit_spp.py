from matplotlib import pyplot as plt
import numpy as np
import os
from pathlib import Path
import pickle
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import ExperimentDetails, fill_nan, nostdout
from postprocessing import postprocess_grads, generate_histograms

from CaVEmain.src.cave import exactConeAlignedCosine
from CaVEmain.src.dataset import optDatasetConstrs
from predmodel import ValueModel
from pyepo.model.grb.shortestpath import shortestPathModel
import pyepo

Path("./results/data/spp").mkdir(parents=True, exist_ok=True)
Path("./results/figures/spp").mkdir(parents=True, exist_ok=True)

NUM_FEATURES = 90
NUM_NODES = 10

DEGREE = 4
EXPERIMENT_SIZE = 20

grads_per_experiment = []
fits_per_experiment = []
alpha_values = np.arange(-1, 1, 0.05)

details = ExperimentDetails(
    experiment_size=10000,
    poly_degree=4,
    problem_type='spp',
    alpha_values=alpha_values,
    inf_rel_norm_stanard_lower=15,
    inf_rel_norm_stanard_upper=17,
    inf_rel_norm_outlier_lower=29,
    inf_rel_norm_outlier_upper=31,
    grads_per_experiment=grads_per_experiment,
    fits_per_experiment=fits_per_experiment
)

for experiment_idx in tqdm(range(details.experiment_size)):
    torch.manual_seed(experiment_idx)

    with nostdout():
        feats, costs = pyepo.data.shortestpath.genData(num_data=1, num_features=120, grid=(6,6), seed=experiment_idx)
        optmodel = shortestPathModel(grid=(6,6))
        dataset = optDatasetConstrs(optmodel, feats, costs)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        cave = exactConeAlignedCosine(optmodel=optmodel, solver="clarabel", processes=4)

    # Estimate gradients with loss functions
    values = []
    gradients = []

    for data in dataloader:
        x, _, _, _, bctr = data
        x = torch.reshape(x, (2, 60))

        for alpha in alpha_values:
            predmodel = ValueModel(alpha=alpha)
            cp = predmodel.forward(x)

            cave_loss = cave(cp, bctr)
            cave_loss.backward(retain_graph=True)
            values.append(cave_loss.item())
            gradients.append(predmodel.alpha.grad.item())

    gradients = fill_nan(gradients)
    grads_per_experiment.append(gradients)
    fit = np.polynomial.polynomial.Polynomial.fit(alpha_values, gradients, details.poly_degree)
    fits_per_experiment.append(fit)

postprocess_grads(details)
generate_histograms(details)
