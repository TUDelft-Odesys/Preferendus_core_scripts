import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from tetra_pfm import TetraSolver

solver = TetraSolver()

# define constants
c1 = 3
c2 = 4
c3 = 7
c4 = 1.2
c5 = 1.7
c6 = 1.9

WT0 = 100


# define objectives
def objective_costs(x):
    x1, x2 = x

    F1 = c2 * x1 + c3 * x2

    return c1 * F1


def objective_wait_time(x):
    x1, x2 = x

    F2 = c5 * x1 + c6 * x2

    return -1 * c4 * F2 + WT0


# run optimizations
bounds = ((1, 5), (3, 8))

result1 = minimize(objective_costs, x0=np.array([1, 1]), bounds=bounds)
result2 = minimize(objective_wait_time, x0=np.array([1, 1]), bounds=bounds)

optimal_result_O1 = result1.fun
optimal_result_O2 = result2.fun

print(f'Objective 1 is minimal for x1 = {result1.x[0]} and x2 = {result1.x[1]}. The costs are then {result1.fun}.')
print(f'Objective 2 is minimal for x1 = {result2.x[0]} and x2 = {result2.x[1]}. The wait time is then {result2.fun}.')

# corner point evaluation with Tetra

min_O1, max_O1 = 75, 228
min_O2, max_O2 = 71.56, 91.12


def preference_P1(variables):
    x1 = variables[:, 0]
    x2 = variables[:, 1]

    costs = objective_costs([x1, x2])
    return 7600 / 51 - 100 * costs / 153


def preference_P2(variables):
    x1 = variables[:, 0]
    x2 = variables[:, 1]

    wait_time = objective_wait_time([x1, x2])
    return 465.849 - 5.11247 * wait_time


alternatives = np.array([
    [1, 3],
    [1, 8],
    [5, 3],
    [5, 8]
])

P1 = preference_P1(alternatives)
P2 = preference_P2(alternatives)

w = [0.5, 0.5]  # weights are equal here
p = [P1, P2]

ret = solver.request(w, p)

# add results to DataFrame and print it
results = np.zeros((4, 3))
results[:, 0] = alternatives[:, 0]
results[:, 1] = alternatives[:, 1]
results[:, 2] = np.multiply(-1, ret)

df = pd.DataFrame(np.round_(results, 2), columns=['x1', 'x2', 'Aggregated preference'])
print(df.to_string())

# plot graphical solution
fig, ax = plt.subplots(figsize=(8, 6))

# Draw constraint lines
ax.vlines(3, -1, 7)
ax.vlines(8, -1, 7)
ax.hlines(1, 1, 10)
ax.hlines(5, 1, 10)

# Draw the feasible region
feasible_set = Polygon(np.array([[3, 1],
                                 [8, 1],
                                 [8, 5],
                                 [3, 5]]),
                       color="lightgrey")
ax.add_patch(feasible_set)

ax.set_xlabel('X2')
ax.set_ylabel('X1')

# Draw the objective function
x2 = np.linspace(1, 6)
ax.plot(x2, (optimal_result_O1 - c1 * c3 * x2) / (c1 * c2), color="purple", label='Costs')
x2 = np.linspace(6, 10)
ax.plot(x2, (WT0 - c4 * c6 * x2 - optimal_result_O2) / (c4 * c5), color="orange", label='Wait time')

ax.scatter([3, 3, 8, 8], [1, 5, 1, 5], marker='*', color='red', label='corner points', s=100)

ax.legend()
ax.grid()
plt.show()
