"""
Python code for example 2 of the addendum: the single stakeholder urban design problem
"""
from scipy.optimize import minimize
import numpy as np

__version__ = 0.1


def objective(variables):
    x1, x2 = variables
    return -1 * (30_000 * x1 + 50_000 * x2)


def constraint(variables):
    x1, x2 = variables
    return -1 * x1 - 2 * x2 + 150


result = minimize(objective, np.array([1, 1]), method='SLSQP', bounds=((0, 60), (0, 50)),
                  constraints={'type': 'ineq', 'fun': constraint})

print(f'The optimal solution is for {round(result.x[0])} houses of type A and '
      f'{round(result.x[1])} houses of type B.')

print(f'This will result in a profit of €{round(-1 * result.fun, 2)}')
