"""
Python code for the single stakeholder urban design problem example
"""
import numpy as np
from scipy.optimize import minimize

"""
In this example we will use the scipy.optimize.minimize function to search for the optimal combination of houses the 
project developer should develop to make the highest profit given all constraints. Since this is a rather simple 
linear problem, we can use a simple and fast solver like the one provided with scipy.

To be able to optimize, we need to define the objective as an python function. Similar we need to define a function for 
the constraint. This is done below, after which the optimization is performed on line 41 and the result is stored to the
variable 'result. This optimal result is also printed on line 54-57.'

For documentation on scipy.optimize.minimize please see the website of scipy or use the help() function.
"""


def objective(variables):
    """
    Objective function that is minimized. Note that the returned value is multiplied by -1. This is since we use a
    minimization algorithm.

    :param variables: list with design variable values
    """
    x1, x2 = variables
    return -1 * (30_000 * x1 + 50_000 * x2)


def constraint(variables):
    """
    Constraint for optimization problem

    :param variables: list with design variable values
    """
    x1, x2 = variables
    return -1 * x1 - 2 * x2 + 150


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

print(f'The optimal solution is for {round(result.x[0])} houses of type A and '
      f'{round(result.x[1])} houses of type B.')

print(f'This will result in a profit of €{round(-1 * result.fun, 2)}')
