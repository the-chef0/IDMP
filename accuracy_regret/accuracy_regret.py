from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np
import os
from pyepo.data import dataset
from pyepo.func import SPOPlus, perturbedFenchelYoung
from pyepo.model.grb.knapsack import knapsackModel
import torch
from torch.utils.data import DataLoader
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_generator import generate_data
from dp.dynamic import DP_Knapsack
from predmodel import ValueModel

from CaVEmain.src.cave import exactConeAlignedCosine
from CaVEmain.src.dataset import optDatasetConstrs

torch.manual_seed(100)

num_items = 100
capacity = 60
runs = 20 # 1 for a single run or a higher number for multiple runs to average

def run(seed=50, graph=True):
    weights, features, values = generate_data(num_items=num_items, capacity=capacity, seed=seed)

    optmodel = knapsackModel(weights=weights, capacity=capacity)
    dataset = optDatasetConstrs(optmodel, features, values)
    #dataset = dataset.optDataset(model=optmodel, feats=features, costs=values)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

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

    arrows = dp_model.calculate_arrows() #Change for the calculate_arrows function in the problem to analyze

    # Estimate gradients with loss functions
    spop_values = []
    spop_gradients = []
    pfy_values = []
    pfy_gradients = []
    cave_values = []
    cave_gradients = []

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

            #w_cave = torch.unsqueeze(w, dim=0)
            cave_loss = cave(cp, bctr)
            cave_loss.backward(retain_graph=True)
            cave_values.append(cave_loss.item())
            cave_gradients.append(predmodel.alpha.grad.item())

    linear_plots, horzizontal_plots, loss_plots, intervals = dp_model.plot(linear=True, horizontal=True, loss=True, z=torch.squeeze(z, dim=0).item())
    
    accuracy_spo = 0
    accuracy_pfyl = 0
    accuracy_cave = 0
    total_points = 0

    regret_spo = 1e9
    pfy_min_regret = 1e9
    cave_min_regret = 1e9

    for i, a in enumerate(alpha_values):
        for (start, end, color) in arrows:

            if min(start[0], end[0]) <= a < max(start[0], end[0]):
                if (spop_gradients[i] < 0 and color[2] == 1.0) or (spop_gradients[i] > 0 and color[0] == 1.0):
                    accuracy_spo += 1
                if (pfy_gradients[i] < 0 and color[2] == 1.0) or (pfy_gradients[i] > 0 and color[0] == 1.0):
                    accuracy_pfyl += 1  
                if (cave_gradients[i] < 0 and color[2] == 1.0) or (cave_gradients[i] > 0 and color[0] == 1.0):
                    accuracy_cave += 1
                total_points += 1
                break
            else:
                continue

        # Regret calculation
        # Ensure that we don't exceed bounds when looking ahead by one
        if i + 1 < len(alpha_values):
            # Check whether we cross zero around the current alpha
            # The gradient can cross zero by either decreasing or increasing
            if (spop_gradients[i] >= 0 and spop_gradients[i+1] <= 0) \
            or (spop_gradients[i] <= 0 and spop_gradients[i+1] >= 0):

                # Find the interval that corresponds to the current alpha
                # I feel like there is a more efficient way to do this, implement if know
                for interval_idx, interval in enumerate(intervals):
                    interval_lower = interval[0]
                    interval_upper = interval[1]
                    
                    if a >= interval_lower and a <= interval_upper:
                        value_at_zero_grad = horzizontal_plots[interval_idx][0]
                        regret_spo = (z - value_at_zero_grad)/z
                        #print(f"Achieved value of {value_at_zero_grad} with regret {regret}")
                        break

            if (pfy_gradients[i] >= 0 and pfy_gradients[i+1] <= 0) \
            or (pfy_gradients[i] <= 0 and pfy_gradients[i+1] >= 0):

                # Find the interval that corresponds to the current alpha
                # I feel like there is a more efficient way to do this, implement if know
                for interval_idx, interval in enumerate(intervals):
                    interval_lower = interval[0]
                    interval_upper = interval[1]
                    
                    if a >= interval_lower and a <= interval_upper:
                        value_at_zero_grad = horzizontal_plots[interval_idx][0]
                        regret_pfy = z - value_at_zero_grad
                        if regret_pfy < pfy_min_regret:
                            pfy_min_regret = regret_pfy
                        #print(f"Achieved value of {value_at_zero_grad} with regret {regret}")
                        break

            if (cave_gradients[i] >= 0 and cave_gradients[i+1] <= 0) \
            or (cave_gradients[i] <= 0 and cave_gradients[i+1] >= 0):

                # Find the interval that corresponds to the current alpha
                # I feel like there is a more efficient way to do this, implement if know
                for interval_idx, interval in enumerate(intervals):
                    interval_lower = interval[0]
                    interval_upper = interval[1]
                    
                    if a >= interval_lower and a <= interval_upper:
                        value_at_zero_grad = horzizontal_plots[interval_idx][0]
                        regret_cave = z - value_at_zero_grad
                        if regret_cave < cave_min_regret:
                            cave_min_regret = regret_cave
                        #print(f"Achieved value of {value_at_zero_grad} with regret {regret}")
                        break

    accuracy_spo = accuracy_spo / total_points
    accuracy_pfyl = accuracy_pfyl / total_points
    accuracy_cave = accuracy_cave / total_points

    regret_pfy = pfy_min_regret/z
    regret_cave = cave_min_regret/z

    if graph:
        # Plot loss function gradients
        # Create base plot with DP solutions
        plt.grid(True)
        plt.xlabel("Alpha")
        plt.ylabel("loss")
        # Plot SPO loss
        spop_loss_plot = plt.plot(alpha_values, spop_values, color="green", label='SPO+')
        #cave_loss_plot = plt.plot(alpha_values, cave_values, color='magenta', label='CaVE')
        pfyl_loss_plot = plt.plot(alpha_values, pfy_values, color='blue', label='PFYL')
        for interval, loss in zip(intervals, loss_plots):
            plt.plot(interval, loss, '--', color='red',)
        plt.legend(['SPO+', 'pfyl', 'SPO'])
        plt.savefig("true_spo_loss.png")

        plt.clf()

        spop_loss_plot = plt.plot(alpha_values, spop_gradients, color="green", label='SPO+')
        cave_loss_plot = plt.plot(alpha_values, cave_gradients, color='magenta', label='CaVE')
        pfyl_loss_plot = plt.plot(alpha_values, pfy_gradients, color='blue', label='PFYL')
        for interval, loss in zip(intervals, horzizontal_plots):
            plt.plot(interval, loss, '--', color='red',)
        plt.legend(['SPO+', 'CaVE', 'pfyl', 'DP'])
        plt.savefig("gradients.png")

    return accuracy_spo, accuracy_pfyl, accuracy_cave, regret_spo, regret_pfy, regret_cave

if runs == 1:
    acumulated_spo, acumulated_pfyl, acumulated_cave, acumulated_regret_spo, acumulated_regret_pfy, acumulated_regret_cave = run(seed=71)
elif runs > 1:
    acumulated_spo = 0
    acumulated_pfyl = 0
    acumulated_cave = 0
    acumulated_regret_spo = 0
    acumulated_regret_pfy = 0
    acumulated_regret_cave = 0
    for i in range(runs):
        accuracy_spo, accuracy_pfyl, accuracy_cave, regret_spo, regret_pfy, regret_cave = run(seed=i, graph=False)
        acumulated_spo += accuracy_spo
        acumulated_pfyl += accuracy_pfyl
        acumulated_cave += accuracy_cave
        acumulated_regret_spo += regret_spo
        acumulated_regret_pfy += regret_pfy
        acumulated_regret_cave += regret_cave
        print(acumulated_spo)


print("The Accuracy for SPO+ is "+str((acumulated_spo/runs)*100)+"%")
print("The Accuracy for PFYL is "+str((acumulated_pfyl/runs)*100)+"%")
print("The Accuracy for CaVE is "+str((acumulated_cave/runs)*100)+"%")
print("The regret for SPO+ is "+str((acumulated_regret_spo/runs)*100)+"%")
print("The regret for PFYL is "+str((acumulated_regret_pfy/runs)*100)+"%")
print("The regret for CaVE is "+str((acumulated_regret_cave/runs)*100)+"%")

