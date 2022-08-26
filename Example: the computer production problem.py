"""
Python code for the computer production problem example
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from scipy.optimize import minimize

"""
In this example we will use the scipy.optimize.minimize function to search for the optimal number of computers of each 
type the company should produce to maximize profit. Since this is a rather simple, linear problem, we can use a simple 
and fast solver like the one provided with scipy.

To be able to optimize, we need to define the objective as an python function. Similar we need to define a function for 
the constraint. This is done below, after which the optimization is performed on line 43 and the result is stored to the
variable 'result. This optimal result is also printed on line 56-59.'

For documentation on scipy.optimize.minimize please see the website of scipy or use the help() function.
"""


def objective(variables):
    """
    Objective function to maximize the profit. Note that the returned value is multiplied by -1. This is since we use a
    minimization algorithm.

    :param variables: list with design variable values
    """
    x1, x2 = variables
    return -1 * (300 * x1 + 500 * x2)


def constraint(variables):
    """
    Constraint for optimization problem

    :param variables: list with design variable values
    """
    x1, x2 = variables
    return -1 * x1 - 2 * x2 + 120


result = minimize(fun=objective, x0=np.array([1, 1]), method='SLSQP', bounds=((0, 60), (0, 50)),
                  constraints={'type': 'ineq', 'fun': constraint})

"""
Explanation of arguments:

fun: function to minimize, here: objective function
x0: initial guesses for the design variables x1 and x2
method: optimization method. The specified method allows for both bounds and constraints.
bounds: boundary values for design variables x1 and x2
constraints: dictionary that contains the type of constraint and the constraint function.
"""

print(f'The optimal solution is for producing {round(result.x[0])} basic computers and '
      f'{round(result.x[1])} advanced computers.')

print(f'This will result in a profit of €{round(-1 * result.fun, 2)}')

"""
The solution of this example can also be found graphically. This figure is created below. You are not required to 
understand the code below, so it is not annotated further.
"""

fig, ax = plt.subplots(figsize=(8, 6))
ax.grid()

# Draw constraint lines
ax.hlines(0, -1, 60)
ax.vlines(0, -1, 120)
ax.plot(np.linspace(-1, 80, 100) * 0 + 50, np.linspace(-1, 120, 100), color="blue")
ax.plot(np.linspace(-1, 60, 100), np.linspace(-1, 60, 100) * 0 + 60, color="blue")
ax.plot(np.linspace(-1, 61, 100), -2 / 1 * np.linspace(-1, 61, 100) + 120, color="blue")

# Draw the feasible region
feasible_set = Polygon(np.array([[0, 0],
                                 [0, 60],
                                 [30, 60],
                                 [50, 20],
                                 [50, 0]]),
                       color="lightgrey")
ax.add_patch(feasible_set)

# Draw the objective function
ax.plot(np.linspace(-1, 61, 100), -3 / 5 * np.linspace(-1, 61, 100) + 78, color="orange")
ax.plot(np.linspace(-1, 37, 100), -3 / 5 * np.linspace(-1, 37, 100) + 20, color="orange")
ax.plot(np.linspace(-1, 61, 100), -3 / 5 * np.linspace(-1, 61, 100) + 50, color="orange")
ax.arrow(-2, 30, 0, 40, width=0.05, head_width=1, head_length=2, color="orange")
ax.text(52, 104, r"$1x_1 + 0x_2 \leq 50$", size=12)
ax.text(52, 64, r"$0x_1 + 1x_2 \leq 60$", size=12)
ax.text(12, 102, r"$1x_1 + 2x_2 \leq 120$", size=12)
ax.text(6, 20, r"$z = 300x_1 + 500x_2$", size=12)
ax.text(15, 48, "solution space", size=12)

# Draw the optimal solution
ax.plot(result.x[1], result.x[0], "*", color="black")
ax.text(32, 64, "Optimal Solution", size=12)
plt.xlabel(r"$x_2$ Deluxe model")
plt.ylabel(r"$x_1$ Standard model")
plt.show()
