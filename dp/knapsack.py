import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

projects = [
    {"novelty": -1, "experience": 10, "cost": 2, "value": 14},
    {"novelty": 1, "experience": 2, "cost": 1, "value": 11},
    {"novelty": -0.5, "experience": 5, "cost": 1, "value": 12},
    {"novelty": 2, "experience": -5, "cost": 1, "value": 10},
]

capacity = 2


# def calculate_value(project, alpha):
#     return project["novelty"] * alpha + project["experience"]


def knapsack(projects, capacity, alpha):
    n = len(projects)
    # Initialize DP table
    dp = np.zeros((n + 1, capacity + 1))
    val = np.zeros((n + 1, capacity + 1))

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if projects[i - 1]["cost"] <= w:
                dp[i][w] = max(
                    dp[i - 1][w],
                    dp[i - 1][w - projects[i - 1]["cost"]]
                    + calculate_value(projects[i - 1], alpha),
                )
                if dp[i - 1][w] > dp[i - 1][
                    w - projects[i - 1]["cost"]
                ] + calculate_value(projects[i - 1], alpha):
                    val[i][w] = val[i - 1][w]
                else:
                    val[i][w] = (
                        val[i - 1][w - projects[i - 1]["cost"]]
                        + projects[i - 1]["value"]
                    )
            else:
                dp[i][w] = dp[i - 1][w]
                val[i][w] = val[i - 1][w]

    return dp[n][capacity], val[n][capacity]


# # Vary alpha and collect results
# alphas = np.linspace(-2, 6, 100)
# values = [knapsack(projects, capacity, alpha)[0] for alpha in alphas]
# true_values = [knapsack(projects, capacity, alpha)[1] for alpha in alphas]
# # Plot the results
# plt.plot(alphas, values)
# plt.plot(alphas, true_values)
# plt.xlabel("Alpha")
# plt.ylabel("Total Value")
# plt.title("Knapsack Value vs. Alpha")
# plt.grid(True)
# plt.savefig("knapsack_value_vs_alpha.png")


# Try 2


class func:
    def __init__(self, slope, intercept, items=[]):
        self.slope = slope
        self.intercept = intercept
        self.items = items


class space:
    def __init__(self, intervals=[(-10, 10)], funcs=[func(0, 0)]):
        self.intervals = intervals
        self.funcs = funcs

    def __str__(self):
        res = ""
        for idx, interval in enumerate(self.intervals):
            func = self.funcs[idx]
            res += f"<{interval[0]} - {interval[1]}>, func: {func.slope}x + {func.intercept}, items: {func.items}\n"
        return res

class magic:
    
    def max(space1, space2):
        inter_list = magic.intersect(space1, space2)
        res_intervals = []
        res_funcs = []
        for interval in inter_list:
            func1 = magic.func_at_interval(space1, interval)
            func2 = magic.func_at_interval(space2, interval)
            # get intersect
            m = magic.intersect_point(func1, func2, interval)
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
            if len(res_intervals)>=2 and res_funcs[-2].slope == res_funcs[-1].slope and res_funcs[-2].intercept == res_funcs[-1].intercept:
                new_interval = (res_intervals[-2][0], res_intervals[-1][1])
                del res_intervals[(len(res_intervals)-1)]
                del res_funcs[(len(res_funcs)-1)]

                del res_intervals[(len(res_intervals)-1)]
                res_intervals.append(new_interval)
        space3 = space(res_intervals, res_funcs)
        print(f'sapce1: {space1} space2: {space2} space3: {space3}')
        return space3

    def intersect(space1, space2):
        i = 0
        j = 0
        ni = len(space1.intervals)
        nj = len(space2.intervals)
        inter_list = []
        # add first interval to inter_list
        front = -10
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


def funcSac(projects, capacity):
    n = len(projects)
    dp = np.empty((n + 1, capacity + 1), dtype=object)
    dp[0, :] = space()
    dp[:, 0] = space()

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if projects[i-1]["cost"] <= w:
                dp[i][w] = magic.max(
                    dp[i - 1][w],
                    magic.add(
                        dp[i - 1][w - projects[i-1]["cost"]],
                        func(projects[i-1]["novelty"], projects[i-1]["experience"]),
                        i-1,
                    ),
                )
            else:
                dp[i][w] = dp[i - 1][w]
    return dp


res = funcSac(projects, capacity)[4][2]
print(res)

for interval, function in zip(res.intervals, res.funcs):
    plt.plot([interval[0], interval[1]], [function.slope * interval[0] + function.intercept, function.slope * interval[1] + function.intercept], '-', color="blue")
    plt.plot([interval[0], interval[1]], [sum([projects[idx]['value'] for idx in function.items])] * 2, '-', color="orange")
plt.savefig("knapsack_value_vs_alpha.png")
