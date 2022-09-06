"""
Python code for the single stakeholder urban design problem example
"""

import numpy as np
from scipy.optimize import LinearConstraint, milp

"""
In this example we will use the scipy.optimize.milp function to search for the optimal number of computers of each 
type the company should produce to maximize profit. Since this is a rather simple, linear problem, we can use a simple 
and fast solver like the one provided with scipy. Also, this algorithm can handle mixed integer problems, like the one
provided here.

For documentation on scipy.optimize.milp please see the website of scipy or use the help() function.
"""

# first, define the objective function. Since it is linear, we can just provide the coefficients with which x1 and x2
# are multiplied. Note the -1: we need to maximize, however, milp is a minimization algorithm!
eq = -1 * np.array([30_000, 50_000])

# next, define the constraints. For this we first provide a matrix A with all the coefficients x1 and x2 are multiplied.
A = np.array([[1, 0], [0, 1], [1, 2]])

# next we determine the upper bounds as vectors
b_u = np.array([60, 50, 150])

# finally, we need to define the lower bound. In our case, these are taken as 0
b_l = np.full_like(b_u, 0)

# we can now define the LinearConstraint
constraints = LinearConstraint(A, b_l, b_u)

# by the integrality array, we tell the algorithm it should take the variables as integers.
# next we can run the optimization
integrality = np.ones_like(eq)

"""Run the optimization"""

result = milp(c=eq, constraints=constraints, integrality=integrality)

print(f'The optimal solution is for {round(result.x[0])} houses of type A and '
      f'{round(result.x[1])} houses of type B.')

print(f'This will result in a profit of €{round(-1 * result.fun, 2)}')
