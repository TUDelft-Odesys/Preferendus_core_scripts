import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

"""
Note that the objectives are programmed as the square root of the actual calculation. This is to improve convergence
with minimize. See also: 
https://stackoverflow.com/questions/47443122/pythons-scipy-optimize-minimize-with-slsqp-fails-with-positive-directional-der
"""


def objective_profit(variables):
    """
    Objective to maximize the profit. Note that the returned value is multiplied by -1. This is since we use a
    minimization algorithm.

    :param variables: list with design variable values
    """
    x1 = variables[0]
    x2 = variables[1]
    x3 = variables[2]

    floor_area = x1 * x2 * x3 / 3
    facade_area = (x1 * x2) * 2 * x3

    revenues = floor_area * 2500
    costs = facade_area * 350 + floor_area * 150

    return -1 * np.sqrt(revenues - costs)


def objective_floor_area(variables):
    """
    Objective to minimize the CO2 emissions.

    :param variables: list with design variable values
    """
    x1 = variables[0]
    x2 = variables[1]
    x3 = variables[2]
    return -1 * np.sqrt(x1 * x2 * x3 / 3)


def constraint_1(variables):
    """
    Constraint for optimization problem

    :param variables: list with design variable values
    """
    x1 = variables[0]
    x2 = variables[1]

    return 25000 - x1 * x2


def constraint_2(variables):
    """
    Constraint for optimization problem

    :param variables: list with design variable values
    """
    return -1 * objective_profit(variables)


# set bounds for all variables
b1 = (0, 200)  # x1
b2 = (0, 200)  # x2
b3 = (0, 50)  # x2
bounds = (b1, b2, b3)

cons = [{'type': 'ineq', 'fun': constraint_1}, {'type': 'ineq', 'fun': constraint_2}]

result1 = minimize(objective_profit, x0=np.array([50, 120, 20]), bounds=bounds, method='SLSQP',
                   constraints=cons)
result2 = minimize(objective_floor_area, x0=np.array([70, 130, 20]), bounds=bounds,
                   constraints=cons)

if result1.success:
    print(
        f'Objective 1 is optimal for x1 = {round(result1.x[0], 2)}, x2 = {round(result1.x[1], 2)} and '
        f'x3 = {round(result1.x[2], 2)}. The profit is then {round(result1.fun ** 2, 2)}.')
else:
    print('Problem with optimization (result1), cannot find a solution')
    print(result1)

if result2.success:
    print(
        f'Objective 2 is optimal for x1 = {round(result2.x[0], 2)}, x2 = {round(result2.x[1], 2)} and x3 = '
        f'{round(result2.x[2], 2)}. The floor area is then {round(result2.fun ** 2, 2)}.')
else:
    print('Problem with optimization (result2), cannot find a solution')
    print(result2)
