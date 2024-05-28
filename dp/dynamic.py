import pyepo
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colormaps
import torch


# # generate data for 2D knapsack
# m = 100  # number of items
# n = 1  # number of data????????????????????????????
# p = 2 * m  # size of feature
# deg = 2  # polynomial degree
# dim = 1  # dimension of knapsack
# noise_width = 0.5  # noise half-width
# caps = [10] * dim  # capacity
# weights, x, c = pyepo.data.knapsack.genData(
#     n, p, m, deg=deg, dim=dim, noise_width=noise_width, seed=36
# )
# # x = x.reshape((m, -1))
# print("AAAAAAAAAAAAA", x.shape)
# x = torch.reshape(torch.Tensor(x), (m, -1))

# capacity = caps[0]


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


class DP_Knapsack:
    def __init__(self, weights, x, c, capacity, left_bound, right_bound):

        self.weights = weights
        self.x = x
        self.c = c
        self.capacity = capacity
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.dp = np.empty((self.x.shape[0] + 1, capacity + 1), dtype=object)
        self.result = space()

    def get_result(self):
        return self.result

    def get_y_values(self):
        return [
            sum([self.c[0][idx] for idx in function.items])
            for function in self.result.funcs
        ]

    def max(space1, space2):
        inter_list = DP_Knapsack.intersect(space1, space2)
        res_intervals = []
        res_funcs = []
        for interval in inter_list:
            func1 = DP_Knapsack.func_at_interval(space1, interval)
            func2 = DP_Knapsack.func_at_interval(space2, interval)
            # get intersect
            m = DP_Knapsack.intersect_point(func1, func2, interval)
            if not m:
                # add interval

                # add highest function at that interval
                if (
                    func1.slope * (interval[0] + 1e-5) + func1.intercept
                    > func2.slope * (interval[0] + 1e-5) + func2.intercept
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
                    > func2.slope * (interval[0] + 1e-5) + func2.intercept
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

    def intersect(space1, space2):
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

    def func_at_interval(space, interval):
        for idx, inter in enumerate(space.intervals):
            if inter[0] <= interval[0] and inter[1] >= interval[1]:
                return space.funcs[idx]

    def intersect_point(func1, func2, interval):
        if func1.slope == func2.slope:
            return None
        m = (func2.intercept - func1.intercept) / (func1.slope - func2.slope)
        if m <= interval[0] or m >= interval[1]:
            return None
        else:
            return m

    def add(space1, func1, item):
        inter_list = space1.intervals
        res_func = []
        for space_function in space1.funcs:
            a = space_function.slope + func1.slope
            b = space_function.intercept + func1.intercept
            items = space_function.items.copy()
            items.append(item)
            res_func.append(func(a, b, items))
        return space(inter_list, res_func)

    def plot(self, linear=True, horizontal=True):
        linear_plots = []
        horizontal_plots = []

        for interval, function in zip(self.result.intervals, self.result.funcs):
            if linear:
                linear_plot = plt.plot(
                    [interval[0], interval[1]],
                    [
                        function.slope * interval[0] + function.intercept,
                        function.slope * interval[1] + function.intercept,
                    ],
                    "--",
                    color="red",
                )

                linear_plots.append(linear_plot[0])

            if horizontal:
                horizontal_plot = plt.plot(
                    [interval[0], interval[1]],
                    [sum([self.c[0][idx] for idx in function.items])] * 2,
                    "--",
                    color="red",
                )

                horizontal_plots.append(horizontal_plot[0])

        return tuple(linear_plots), tuple(horizontal_plots)

    def solve(self):
        n = self.x.shape[0]
        dp = np.empty((n + 1, self.capacity + 1), dtype=object)
        dp[0, :] = space(intervals=[(self.left_bound, self.right_bound)])
        dp[:, 0] = space(intervals=[(self.left_bound, self.right_bound)])
        int_weights = np.ceil(self.weights).astype("int")
        for i in range(1, n + 1):
            for w in range(self.capacity + 1):
                if self.weights[i - 1] <= w:
                    dp[i][w] = DP_Knapsack.max(
                        dp[i - 1][w],
                        DP_Knapsack.add(
                            dp[i - 1][w - int_weights[i - 1]],
                            func(self.x[i - 1][0], self.x[i - 1][1]),
                            i - 1,
                        ),
                    )
                else:
                    dp[i][w] = dp[i - 1][w]
        self.dp = dp
        self.result = dp[n][self.capacity]
        return dp[n][self.capacity]


# dp_problem = DP_Knapsack(weights[0], x, c, -6, 6)
# dp_problem.solve()
# print(dp_problem.result)
# dp_problem.plot()
# plt.show()
# print(dp_problem.get_result().intervals)
# print(dp_problem.get_y_values())

# def calculate_value(x, alpha):
#     return x[0] * alpha + x[1]


# def knapsack(weights, x, c, alpha):
#     n = x.shape[0]
#     # Initialize DP table
#     dp = np.zeros((n + 1, capacity + 1))
#     val = np.zeros((n + 1, capacity + 1))
#     int_weights = np.ceil(weights).astype('int')
#     for i in range(1, n + 1):
#         for w in range(capacity + 1):
#             if weights[i - 1] <= w:
#                 dp[i][w] = max(dp[i - 1][w],
#                                dp[i - 1][w - int_weights[i - 1]] + calculate_value(x[i - 1], alpha))
#                 if dp[i - 1][w] > dp[i - 1][w - int_weights[i - 1]] + calculate_value(x[i - 1], alpha):
#                     val[i][w] = val[i - 1][w]
#                 else:
#                     val[i][w] = val[i - 1][w - int_weights[i - 1]] + c[i - 1]
#             else:
#                 dp[i][w] = dp[i - 1][w]
#                 val[i][w] = val[i - 1][w]

#     return dp[n][capacity], val[n][capacity]


# Vary alpha and collect results
# alphas = np.linspace(-2, 6, 1000)
# values = [knapsack(weights[0], x, c[0], alpha)[0] for alpha in alphas]
# true_values = [knapsack(weights[0], x, c[0], alpha)[1] for alpha in alphas]
# # Plot the results
# plt.plot(alphas, values)
# plt.plot(alphas, true_values)
# plt.xlabel("Alpha")
# plt.ylabel("Total Value")
# plt.title("Knapsack Value vs. Alpha")
# plt.grid(True)
# plt.savefig("knapsack_value_vs_alpha.png")
