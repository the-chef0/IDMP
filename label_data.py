from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np
from pyepo.data import dataset
from pyepo.func import SPOPlus, perturbedFenchelYoung
import torch
from torch.utils.data import DataLoader

from data_generator import generate_data
from dp.SP_dynamic import SP_dynamic
from predmodel import ValueModel
from pyepo.model.grb.shortestpath import shortestPathModel
import pyepo.data.shortestpath
from CaVEmain.src.cave import exactConeAlignedCosine

experiment_size = 2000
num_data = 1  # number of data
grid = (5, 5)  # grid size
num_feat = 2 * ((grid[0] - 1) * grid[1] + (grid[1] - 1) * grid[0])  # size of feature
alpha_range = [-15, 15]
alpha_values = np.arange(alpha_range[0], alpha_range[1], 0.05)

left_count = 0
left_array = []
right_count = 0
right_array = []
middle_count = 0
middle_array = []

s_inter_array = []
s_inter_count = 0
m_inter_array = []
m_inter_count = 0
l_inter_array = []
l_inter_count = 0

tiny_inter_array = []
tiny_inter_count = 0
for exp in range(experiment_size):
    print(exp, "/", experiment_size)
    torch.manual_seed(exp)

    x, c = pyepo.data.shortestpath.genData(
        num_data, num_feat, grid, deg=1, noise_width=0, seed=exp
    )
    optmodel = shortestPathModel(grid=grid)
    data = dataset.optDataset(model=optmodel, feats=x, costs=c)
    dataloader = DataLoader(data, batch_size=1, shuffle=True)

    x = x.reshape((2, -1))
    sp_dynamic = SP_dynamic(x, c, grid, alpha_range[0], alpha_range[1])
    sp_dynamic.solve()

    # label solve:
    # to -inf
    # to +inf
    # middle
    # -1,1
    # -4, 4
    # tiny max interval

    label = ""
    y_values = sp_dynamic.get_y_values()
    min_value_index = np.argmin(y_values)
    min_interval = sp_dynamic.result.intervals[min_value_index]
    if min_value_index == 0:
        label = "Left"
        left_count += 1
        left_array.append(
            {
                "label": label,
                "grid_size": grid,
                "seed": exp,
                "alpha": alpha_range,
            }
        )
    elif (
        min_value_index == len(y_values) - 1
        or y_values[min_value_index] == y_values[-1]
    ):
        label = "Right"
        right_count += 1
        right_array.append(
            {
                "label": label,
                "grid_size": grid,
                "seed": exp,
                "alpha": alpha_range,
            }
        )
        # dp_model.plot()
        # plt.title(label)
        # plt.show()
    else:
        label = "Middle"
        middle_count += 1
        middle_array.append(
            {
                "label": label,
                "grid_size": grid,
                "seed": exp,
                "alpha": alpha_range,
            }
        )
        # -1, 1
        if min_interval[0] >= -1 and min_interval[1] <= 1:
            label = "small_interval"
            s_inter_count += 1
            s_inter_array.append(
                {
                    "label": label,
                    "grid_size": grid,
                    "seed": exp,
                    "alpha": alpha_range,
                }
            )
        # -4,4
        elif min_interval[0] >= -4 and min_interval[1] <= 4:
            label = "medium_interval"
            m_inter_count += 1
            m_inter_array.append(
                {
                    "label": label,
                    "grid_size": grid,
                    "seed": exp,
                    "alpha": alpha_range,
                }
            )
        # large
        else:
            label = "large_interval"
            l_inter_count += 1
            l_inter_array.append(
                {
                    "label": label,
                    "grid_size": grid,
                    "seed": exp,
                    "alpha": alpha_range,
                }
            )

        # dp_model.plot(horizontal=True, linear=False)
        # plt.title(label)
        # plt.show()
    # tiny interval
    if min_interval[1] - min_interval[0] < (alpha_range[1] - alpha_values[0]) * 0.01:
        label = "tiny_interval"
        tiny_inter_count += 1
        tiny_inter_array.append(
            {
                "label": label,
                "grid_size": grid,
                "seed": exp,
                "alpha": alpha_range,
            }
        )

# print("Before: ", np.array(left_array))
np.save("labled_data\\SP_left_labled_data.npy", left_array)
np.save("labled_data\\SP_right_labled_data.npy", right_array)
np.save("labled_data\\SP_middle_labled_data.npy", middle_array)

np.save("labled_data\\SP_small_labled_data.npy", s_inter_array)
np.save("labled_data\\SP_medium_labled_data.npy", m_inter_array)
np.save("labled_data\\SP_large_labled_data.npy", l_inter_array)

np.save("labled_data\\SP_tiny_labled_data.npy", tiny_inter_array)

print(
    "Left count: ",
    left_count,
    ", Middle count: ",
    middle_count,
    ", Right Count: ",
    right_count,
    ", Small Count: ",
    s_inter_count,
    ", Medium Count: ",
    m_inter_count,
    ", Large Count: ",
    l_inter_count,
    ", Tiny Count: ",
    tiny_inter_count,
)
