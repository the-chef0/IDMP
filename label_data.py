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

experiment_size = 30
num_items = 50
capacity = 30
alpha_range = [-30, 30]
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

    weights, features, values = generate_data(
        num_items=num_items, capacity=capacity, seed=exp
    )

    optmodel = knapsackModel(weights=weights, capacity=capacity)
    data_set = dataset.optDataset(model=optmodel, feats=features, costs=values)
    dataloader = DataLoader(data_set, batch_size=1, shuffle=True)

    # Estimate gradients with dynamic programming
    features = features.reshape((2, num_items))
    dp_model = DP_Knapsack(
        weights[0], features, values, capacity, alpha_values[0], alpha_values[-1]
    )
    dp_model.solve()

    # label solve:
    # to -inf
    # to +inf
    # middle
    # -1,1
    # -4, 4
    # tiny max interval

    label = ""
    y_values = dp_model.get_y_values()
    max_value_index = np.argmax(y_values)
    max_interval = dp_model.result.intervals[max_value_index]
    if max_value_index == 0:
        label = "Left"
        left_count += 1
        left_array.append(
            {
                "label": label,
                "num_items": num_items,
                "capacity": capacity,
                "seed": exp,
                "alpha": alpha_range,
            }
        )
    elif (
        max_value_index == len(y_values) - 1
        or y_values[max_value_index] == y_values[-1]
    ):
        label = "Right"
        right_count += 1
        right_array.append(
            {
                "label": label,
                "num_items": num_items,
                "capacity": capacity,
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
                "num_items": num_items,
                "capacity": capacity,
                "seed": exp,
                "alpha": alpha_range,
            }
        )
        # -1, 1
        if max_interval[0] >= -1 and max_interval[1] <= 1:
            label = "small_interval"
            s_inter_count += 1
            s_inter_array.append(
                {
                    "label": label,
                    "num_items": num_items,
                    "capacity": capacity,
                    "seed": exp,
                    "alpha": alpha_range,
                }
            )
        # -4,4
        elif max_interval[0] >= -4 and max_interval[1] <= 4:
            label = "medium_interval"
            m_inter_count += 1
            m_inter_array.append(
                {
                    "label": label,
                    "num_items": num_items,
                    "capacity": capacity,
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
                    "num_items": num_items,
                    "capacity": capacity,
                    "seed": exp,
                    "alpha": alpha_range,
                }
            )

        # dp_model.plot(horizontal=True, linear=False)
        # plt.title(label)
        # plt.show()
    # tiny interval
    if max_interval[1] - max_interval[0] < (alpha_range[1] - alpha_values[0]) * 0.01:
        label = "tiny_interval"
        tiny_inter_count += 1
        tiny_inter_array.append(
            {
                "label": label,
                "num_items": num_items,
                "capacity": capacity,
                "seed": exp,
            }
        )

# print("Before: ", np.array(left_array))
np.save("labled_data\\left_labled_data.npy", left_array)
np.save("labled_data\\right_labled_data.npy", right_array)
np.save("labled_data\\middle_labled_data.npy", middle_array)

np.save("labled_data\\small_labled_data.npy", s_inter_array)
np.save("labled_data\\medium_labled_data.npy", m_inter_array)
np.save("labled_data\\large_labled_data.npy", l_inter_array)

np.save("labled_data\\tiny_labled_data.npy", tiny_inter_array)

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
