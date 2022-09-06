"""
Python code for the multi-stakeholder urban design problem example
"""

import numpy as np
from scipy.optimize import milp, LinearConstraint

"""
In this example we will use the scipy.optimize.milp function to search for the optimal number of computers of each 
type the company should produce to maximize profit. Since this is a rather simple, linear problem, we can use a simple 
and fast solver like the one provided with scipy. Also, this algorithm can handle mixed integer problems, like the one
provided here.

For documentation on scipy.optimize.milp please see the website of scipy or use the help() function.
"""

# first, define the objective function. Since it is linear, we can just provide the coefficients with which x1 and x2
# are multiplied. Note the -1: we need to maximize, however, milp is a minimization algorithm!
eq_proj_dev = -1 * np.array([11125, 16650, 16500, 11250, 16950, 21700])
eq_municipal = -1 * np.array([1, 0, 0, 1, 0, 0])

# next, define the constraints. For this we first provide a matrix A with all the coefficients x1 and x2 are multiplied.
A = np.array([
    [-1, -1, -1, -1, -1, -1],
    [1, 1, 1, 1, 1, 1],

    # x1
    [-0.8, 0.2, 0.2, 0.2, 0.2, 0.2],
    [0.7, -0.3, -0.3, -0.3, -0.3, -0.3],
    # x2
    [0.15, -0.85, 0.15, 0.15, 0.15, 0.15],
    [-0.25, 0.75, -0.25, -0.25, -0.25, -0.25],
    # x3
    [0.1, 0.1, -0.9, 0.1, 0.1, 0.1],
    [-0.15, -0.15, 0.85, -0.15, -0.15, -0.15],
    # x4
    [0.15, 0.15, 0.15, -0.85, 0.15, 0.15],
    [-0.2, -0.2, -0.2, 0.8, -0.2, -0.2],
    # x5
    [0.1, 0.1, 0.1, 0.1, -0.9, 0.1],
    [-0.2, -0.2, -0.2, -0.2, 0.8, -0.2],
    # x6
    [0.05, 0.05, 0.05, 0.05, 0.05, -0.95],
    [-0.1, -0.1, -0.1, -0.1, -0.1, 0.9],
])

# next we determine the upper bounds as vectors
b_u = np.array([-200, 260, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# finally, we need to define the lower bound. In our case, these are taken as -inf
b_l = np.full_like(b_u, -np.inf)

# we can now define the LinearConstraint
constraints = LinearConstraint(A, b_l, b_u)

# by the integrality array, we tell the algorithm it should take the variables as integers.
integrality = np.ones_like(eq_proj_dev)

"""Run the optimizations"""

# below, the project developer objective is optimized and the result is printed.
print('Optimizing on the project developer’s objective:')
result = milp(c=eq_proj_dev, constraints=constraints, integrality=integrality)

types = ['A', 'C', 'L', 'M', 'Q', 'S']  # make a list that contains the names of all types of houses
print('The optimal solution is for:')
for i, n in enumerate(result.x):
    print(f'\t{round(n)} houses of type {types[i]}')

print(f'This will result in a profit of €{round(-1 * result.fun, 2)}')

# secondly, the municipality's objective is optimized
print()
print('Optimizing on the municipality’s objective:')
result = milp(c=eq_municipal, constraints=constraints, integrality=integrality)

print('The optimal solution is for:')
for i, n in enumerate(result.x):
    print(f'\t{round(n)} houses of type {types[i]}')

print(f'the maximum number of affordable houses is {round(-1 * result.fun)}')

"""
Lastly, the objective for the project developer is optimized again, only now the municipality's objective is added as 
a constraint. This constraint is added to the previously declared arrays by using the append method
"""

print()
print('Optimizing using the constraint method:')
A = np.append(A, [[-1, 0, 0, -1, 0, 0]], axis=0)
b_u = np.append(b_u, -130)
b_l = np.full_like(b_u, -np.inf)
constraints = LinearConstraint(A, b_l, b_u)

result = milp(c=eq_proj_dev, constraints=constraints, integrality=integrality)

print('The optimal solution is for:')
for i, n in enumerate(result.x):
    print(f'\t{round(n)} houses of type {types[i]}')

print(f'This will result in a profit of €{round(-1 * result.fun, 2)}')
