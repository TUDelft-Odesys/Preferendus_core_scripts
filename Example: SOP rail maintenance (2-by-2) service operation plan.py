import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd

from genetic_algorithm_pfm.tetra_pfm import TetraSolver

solver = TetraSolver()

# define constants
c1 = 50 / 21
c2 = -5 / 21
c3 = 20 / 3
c4 = -8 / 3

a0 = 80

tc_min = 70
av_min = 100


# define objectives
def objective_comfort(x):
    x1, x2 = x
    return -1 * (c1 * x1 + c2 * x2)


def objective_availability(x):
    x1, x2 = x
    return -1 * (a0 - c3 * x1 - c4 * x2)


# run optimizations
bounds = ((1, 52), (10, 100))

constraint_1 = {'type': 'ineq', 'fun': lambda x: c1 * x[0] + c2 * x[1] - tc_min}
constraint_2 = {'type': 'ineq', 'fun': lambda x: a0 - c3 * x[0] - c4 * x[1] - av_min}

result1 = minimize(objective_comfort, x0=np.array([1, 1]), bounds=bounds, constraints=constraint_1)
result2 = minimize(objective_availability, x0=np.array([1, 1]), bounds=bounds, constraints=constraint_2)

optimal_result_O1 = -1 * result1.fun
optimal_result_O2 = -1 * result2.fun

if result1.success:
    print(f'Objective 1 is maximal for x1 = {result1.x[0]} and x2 = {result1.x[1]}. '
          f'The comfort are then {round(-1 * result1.fun, 2)}.')
else:
    print('Problem with optimization (result1), cannot find a solution')
    print(result1)

if result2.success:
    print(f'Objective 2 is maximal for x1 = {result2.x[0]} and x2 = {result2.x[1]}. '
          f'The availability is then {round(-1 * result2.fun, 2)}.')
else:
    print('Problem with optimization (result2), cannot find a solution')
    print(result2)

"""
Below, all corner points are evaluated in Tetra
"""


def preference_P1(variables):
    x1 = variables[:, 0]
    x2 = variables[:, 1]

    comfort = objective_comfort([x1, x2])
    return 25 * comfort / 13 - 1750 / 13


def preference_P2(variables):
    x1 = variables[:, 0]
    x2 = variables[:, 1]

    availability = objective_availability([x1, x2])
    return 5 * availability / 12 - 125 / 3


alternatives = np.array([
    [1, 10],
    [1, 100],
    [52, 10],
    [52, 100]
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
print()

"""
The graphical solution can also be plotted
"""

# plot graphical solution
fig, ax = plt.subplots(figsize=(8, 6))

# Draw constraint lines
ax.vlines(10, -5, 60)
ax.vlines(100, -5, 60)
ax.hlines(1, 0, 110)
ax.hlines(52, 0, 110)

# Draw the feasible region
feasible_set = Polygon(np.array([[10, 1],
                                 [100, 1],
                                 [100, 52],
                                 [10, 52]]),
                       color="lightgrey")
ax.add_patch(feasible_set)

ax.set_xlabel('X2')
ax.set_ylabel('X1')

# Draw the objective function
x2 = np.linspace(0, 30)
ax.plot(x2, (optimal_result_O1 - c2 * x2) / c1, color="purple", label='Comfort')
x2 = np.linspace(80, 110)
ax.plot(x2, (a0 - c4 * x2 - optimal_result_O2) / c3, color="orange", label='Availability')

ax.scatter([10, 10, 100, 100], [1, 52, 1, 52], marker='*', color='red', label='corner points', s=100)

ax.legend()
ax.grid()
plt.show()
