"""
Python code for the bridge design problem (Chapter 5.2 Example 2)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, milp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from genetic_algorithm_pfm.tetra_pfm import TetraSolver
from genetic_algorithm_pfm import GeneticAlgorithm

solver = TetraSolver()

# define constants
c1 = 3  # costs per material
c2 = 4  # material used per metre bridge span
c3 = 7  # material used per metre air draft
c4 = 1.2  # relation between waiting time and traffic flow
c5 = 1.7  # traffic flow per metre bridge span
c6 = 1.9  # traffic flow per metre air draft

WT0 = 100 # minimal waiting time

"""
First we use MILP to solve the problem
"""

# first, define the objective function. Since it is linear, we can just provide the coefficients with which x1 and x2
# are multiplied. Note the -1: we need to maximize, however, milp is a minimization algorithm!
c_costs = 1 * np.array([c1 * c2, c1 * c3])
c_wait_time = -1 * np.array([c4 * c5, c4 * c6])

# next, define the constraints. For this we first provide a matrix A with all the coefficients x1 and x2 are multiplied.
A = np.array([[1, 0], [0, 1]])

# next we determine the upper bounds as vectors
b_u = np.array([5, 8])

# finally, we need to define the lower bound. In our case, these are taken as 0
b_l = np.array([1, 3])

# we can now define the LinearConstraint
constraints = LinearConstraint(A, b_l, b_u)

# the integrality array will tell the algorithm what type of variables (0 = continuous; 1 = integer) there are
integrality = np.zeros_like(c_costs)

# Run the optimization
result1 = milp(c=c_costs, constraints=constraints, integrality=integrality)
result2 = milp(c=c_wait_time, constraints=constraints, integrality=integrality)

print('Results MILP')
print(f'Objective 1 is minimal for x1 = {result1.x[0]} and x2 = {result1.x[1]}. The costs are then {result1.fun}.')
print(f'Objective 2 is minimal for x1 = {result2.x[0]} and x2 = {result2.x[1]}. '
      f'The wait time is then {result2.fun + WT0}.')
print()

"""
Below we will do the same optimisation, but now with minimize from Scipy
"""


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

"""Run the optimization"""
result1 = minimize(objective_costs, x0=np.array([1, 1]), bounds=bounds)
result2 = minimize(objective_wait_time, x0=np.array([1, 1]), bounds=bounds)

optimal_result_O1 = result1.fun
optimal_result_O2 = result2.fun

print('Results Minimize')
print(f'Objective 1 is minimal for x1 = {result1.x[0]} and x2 = {result1.x[1]}. The costs are then {result1.fun}.')
print(f'Objective 2 is minimal for x1 = {result2.x[0]} and x2 = {result2.x[1]}. The wait time is then {result2.fun}.')
print()

"""
Below, all corner points are evaluated in Tetra
"""


def preference_P1(variables):
    x1 = variables[:, 0]
    x2 = variables[:, 1]

    costs = objective_costs([x1, x2])
    return 7600 / 51 - 100 * costs / 153


def preference_P2(variables):
    x1 = variables[:, 0]
    x2 = variables[:, 1]

    wait_time = objective_wait_time([x1, x2])
    return 291.747 - 2.67953 * wait_time


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
print()

"""
The graphical solution can also be plotted
"""

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

"""
The optimisation above initially considers only the single stakeholders and evaluates only the corner points. However, 
as you will in other examples, this is no guarantee that the optimal solution is found. Here, the optimal point is not 
on any of the corners. To find the optimal solutions in these cases, a multi-objective optimisation is performed. 

The same is showed below for the bridge problem, so you can see how the single-objective optimisations, the 
multi-objective evaluation, and the multi-objective optimisation relate to each other.
"""

weights = [0.5, 0.5]


def objective(variables):
    p1 = preference_P1(variables)
    p2 = preference_P2(variables)

    return weights, [p1, p2]


bounds = [[1, 5], [3, 8]]
options = {
    'aggregation': 'tetra',
    "n_pop": 120,
    "max_stall": 60,
    "n_iter": 1000,
    "n_bits": 8,
    "r_cross": 0.9
}

# run the GA and print its result
ga = GeneticAlgorithm(objective=objective, constraints=[], bounds=bounds, options=options)
score, design_variables, _ = ga.run()

print()
print('Results multi-objective optimisation')
print(f'x1 = {design_variables[0]} and x2 = {design_variables[1]}.')

plt.show()
