import numpy as np
from scipy.optimize import minimize

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

    floor_area = x1 * x2 
    facade_area = (x1 + x2) * 2 * x3

    revenues = floor_area * 55
    costs = facade_area * 3.5 + floor_area * 1.5

    return -1 * np.sqrt(revenues - costs)


def objective_energy_use(variables):
    """
    Objective to minimize the energy consumption.
    :param variables: list with design variable values
    """
    x1 = variables[0]
    x2 = variables[1]
    x3 = variables[2]
    energy_costs_per_m3 = 0.32
    return np.sqrt(x1 * x2 * x3 * energy_costs_per_m3)


def constraint_1(variables):
    """
    Constraint for optimization problem
    :param variables: list with design variable values
    """
    x1 = variables[0]
    x2 = variables[1]

    return 35000 - x1 * x2  # the building footprint cannot be bigger than 35,000 m2


def constraint_2(variables):
    """
    Constraint for optimization problem
    :param variables: list with design variable values
    """
    return objective_profit(variables) ** 2 - (3e4)  # minimal profit is 30,000 euros


# set bounds for all variables
b1 = (80, 200)  # x1
b2 = (75, 200)  # x2
b3 = (3, 12)  # x3
bounds = (b1, b2, b3)

cons = [{'type': 'ineq', 'fun': constraint_1}, {'type': 'ineq', 'fun': constraint_2}]

# due to the non-linearity, it might be needed to increase the maximum number of iterations for minimize
# see line 3-4 of this code block
result1 = minimize(objective_profit, x0=np.array([130, 160, 10]), bounds=bounds,
                   constraints=cons, options={'maxiter': 300})
result2 = minimize(objective_energy_use, x0=np.array([70, 130, 20]), bounds=bounds,
                   constraints=cons)

if result1.success:
    print(
        f'Objective 1 is optimal for x1 = {round(result1.x[0], 2)}, x2 = {round(result1.x[1], 2)} and '
        f'x3 = {round(result1.x[2], 2)}. The profit is then €{round(result1.fun ** 2, 2)}.')
else:
    print('Problem with optimization (result1), cannot find a solution')
    print(result1)

if result2.success:
    print(
        f'Objective 2 is optimal for x1 = {round(result2.x[0], 2)}, x2 = {round(result2.x[1], 2)} and x3 = '
        f'{round(result2.x[2], 2)}. The energy costs are then €{round(result2.fun ** 2, 2)}.')
else:
    print('Problem with optimization (result2), cannot find a solution')
    print(result2)
