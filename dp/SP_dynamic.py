import pyepo
import matplotlib
import numpy as np
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from matplotlib import colormaps
import torch
import pyepo.data.shortestpath
from pyepo.model.grb.shortestpath import shortestPathModel
from pyepo.data import dataset
from torch.utils.data import DataLoader


class func:
    def __init__(self, slope, intercept, items=[]):
        self.slope = slope
        self.intercept = intercept
        self.items = items


class space:
    def __init__(self, intervals=[(-100, 100)], funcs=[func(0, 0)]):
        self.intervals = intervals
        self.funcs = funcs

    def __str__(self):
        res = ""
        for idx, interval in enumerate(self.intervals):
            func = self.funcs[idx]
            res += f"<{interval[0]} - {interval[1]}>, func: {func.slope}x + {func.intercept}, items: {func.items}\n"
        return res


class SP_dynamic:
    def __init__(self, features, true_cost, grid, left_bound, right_bound):
        # start 0,0
        # end [grid[0]-1, grid[1]-1]
        self.features = features
        self.true_cost = true_cost
        self.grid = grid
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.arcs = self._getArcs()
        self.dp = np.empty(
            ((self.grid[0]) * (self.grid[0]), len(self.arcs)), dtype=object
        )
        self.result = space()

    def _getArcs(self):
        arcs = []
        for i in range(self.grid[0]):
            # edges on rows
            for j in range(self.grid[1] - 1):
                v = i * self.grid[1] + j
                arcs.append((v, v + 1))
            # edges in columns
            if i == self.grid[0] - 1:
                continue
            for j in range(self.grid[1]):
                v = i * self.grid[1] + j
                arcs.append((v, v + self.grid[1]))
        return arcs

    def min(self, space1, space2):
        inter_list = self.intersect(space1, space2)
        res_intervals = []
        res_funcs = []
        for interval in inter_list:
            func1 = self.func_at_interval(space1, interval)
            func2 = self.func_at_interval(space2, interval)
            m = self.intersect_point(func1, func2, interval)
            if not m:
                # add interval

                # add highest function at that interval
                if (
                    func1.slope * (interval[0] + 1e-5) + func1.intercept
                    < func2.slope * (interval[0] + 1e-5) + func2.intercept
                ):
                    res_func = func1
                else:
                    res_func = func2
                # check if line is the same, if so increase the interval
                res_intervals.append(interval)
                res_funcs.append(res_func)
            else:
                lower_interval = (interval[0], m)
                higher_interval = (m, interval[1])
                if (
                    func1.slope * (interval[0] + 1e-5) + func1.intercept
                    < func2.slope * (interval[0] + 1e-5) + func2.intercept
                ):
                    lower_func = func1
                    higher_func = func2
                else:
                    lower_func = func2
                    higher_func = func1

                res_intervals.append(lower_interval)
                res_funcs.append(lower_func)

                res_intervals.append(higher_interval)
                res_funcs.append(higher_func)
            if (
                len(res_intervals) >= 2
                and res_funcs[-2].slope == res_funcs[-1].slope
                and res_funcs[-2].intercept == res_funcs[-1].intercept
            ):
                new_interval = (res_intervals[-2][0], res_intervals[-1][1])
                del res_intervals[(len(res_intervals) - 1)]
                del res_funcs[(len(res_funcs) - 1)]

                del res_intervals[(len(res_intervals) - 1)]
                res_intervals.append(new_interval)
        space3 = space(res_intervals, res_funcs)
        # print(f'sapce1: {space1} space2: {space2} space3: {space3}')
        return space3

    def intersect(self, space1, space2):
        i = 0
        j = 0
        ni = len(space1.intervals)
        nj = len(space2.intervals)
        inter_list = []
        front = space1.intervals[0][0]
        # merge
        while True:
            if i < ni and j < nj:
                if space1.intervals[i][1] < space2.intervals[j][1]:
                    inter_list.append((front, space1.intervals[i][1]))
                    front = space1.intervals[i][1]
                    i += 1
                elif space1.intervals[i][1] == space2.intervals[j][1]:
                    inter_list.append((front, space1.intervals[i][1]))
                    front = space1.intervals[i][1]
                    i += 1
                    j += 1
                else:
                    inter_list.append((front, space2.intervals[j][1]))
                    front = space2.intervals[j][1]
                    j += 1
            elif i < ni:
                inter_list.append((front, space1.intervals[i][1]))
                front = space1.intervals[i][1]
                i += 1
            elif j < nj:
                inter_list.append((front, space2.intervals[j][1]))
                front = space2.intervals[j][1]
                j += 1
            else:
                break
        # print(inter_list)
        return inter_list

    def func_at_interval(self, space, interval):
        for idx, inter in enumerate(space.intervals):
            if inter[0] <= interval[0] and inter[1] >= interval[1]:
                return space.funcs[idx]

    def intersect_point(self, func1, func2, interval):
        if func1.slope == func2.slope:
            return None
        m = (func2.intercept - func1.intercept) / (func1.slope - func2.slope)
        if m <= interval[0] or m >= interval[1]:
            return None
        else:
            return m

    def add(self, space1, func1, item):
        inter_list = space1.intervals
        res_func = []
        for space_function in space1.funcs:
            a = space_function.slope + func1.slope
            b = space_function.intercept + func1.intercept
            items = space_function.items.copy()
            items.append(item)
            res_func.append(func(a, b, items))
        return space(inter_list, res_func)

    def plot(self, linear=True, horizontal=True, loss=False, z=None):
        linear_plots = []
        horizontal_plots = []
        loss_plots = []
        intervals = []
        for interval, function in zip(self.result.intervals, self.result.funcs):
            intervals.append([interval[0], interval[1]])
            if linear:
                linear_plots.append(
                    [
                        function.slope * interval[0] + function.intercept,
                        function.slope * interval[1] + function.intercept,
                    ]
                )

            if horizontal:
                horizontal_plots.append(
                    [sum([self.true_cost[0][idx] for idx in function.items])] * 2
                )
            if loss and z:
                loss_plots.append(
                    [z - sum([self.true_cost[0][idx] for idx in function.items])] * 2,
                )

        # arrows = self.calculate_arrows()
        # for (start, end, color) in arrows:
        #     plt.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', color=color))

        return linear_plots, horizontal_plots, loss_plots, intervals

    def solve(self):
        dp = np.empty(((self.grid[0]) * (self.grid[0]), len(self.arcs)), dtype=object)

        dp[:, 0] = space(
            intervals=[(self.left_bound, self.right_bound)], funcs=[func(0, 10000)]
        )
        dp[0, :] = space(intervals=[(self.left_bound, self.right_bound)])
        for v in range(1, dp.shape[0]):
            for i in range(1, dp.shape[1]):
                # if v == 2:
                #     return
                curr_best = space(
                    intervals=[(self.left_bound, self.right_bound)],
                    funcs=[func(0, 10000)],
                )
                # print("----------------")
                for idx, arc in enumerate(self.arcs):
                    if arc[1] == v:
                        # print("v: ", v, ", i: ", i, "  arc: ", arc)
                        # print("curr_best: ", curr_best)
                        # print("dp_thing: ", dp[arc[0], i - 1])
                        add = self.add(
                            dp[arc[0], i - 1],
                            func(self.features[0][idx], self.features[1][idx]),
                            idx,
                        )
                        # print("add: ", add)
                        curr_best = self.min(
                            add,
                            curr_best,
                        )
                        # print("min: ", add)

                # rec equ
                # print("less edges: ", dp[v, i - 1])
                dp[v, i] = self.min(dp[v, i - 1], curr_best)
                # print("final: ", dp[v, i])
        self.dp = dp
        self.result = dp[dp.shape[0] - 1][dp.shape[1] - 1]
        return dp[dp.shape[0] - 1][dp.shape[1] - 1]


# Test generation code
# num_data = 1  # number of data
# grid = (5, 5)  # grid size
# num_feat = 2 * ((grid[0] - 1) * grid[1] + (grid[1] - 1) * grid[0])  # size of feature

# x, c = pyepo.data.shortestpath.genData(
#     num_data, num_feat, grid, deg=1, noise_width=0, seed=1
# )
# optmodel = shortestPathModel(grid=grid)
# dataset = dataset.optDataset(model=optmodel, feats=x, costs=c)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
# for data in dataloader:
#     x, c, w, z = data
#     print(z)


# x = x.reshape((2, -1))
# sp_dynamic = SP_dynamic(x, c, grid, -5, 5)
# res = sp_dynamic.solve()
# print(res)


# linear_plots, horzizontal_plots, loss_plots, intervals = sp_dynamic.plot(
#     linear=True, horizontal=True, loss=False
# )

# for interval, loss in zip(intervals, linear_plots):
#     plt.plot(
#         interval,
#         loss,
#         "--",
#         color="red",
#     )
# for interval, loss in zip(intervals, horzizontal_plots):
#     plt.plot(
#         interval,
#         loss,
#         "--",
#         color="red",
#     )
#     print(loss)

# plt.show()
