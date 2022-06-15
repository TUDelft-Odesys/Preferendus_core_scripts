"""
Python code for example 3 of the addendum: The multi-stakeholder urban design problem
"""
import numpy as np
from scipy.optimize import minimize

__version__ = 0.1


def objective_project_dev(variables):
    x1, x2, x3, x4, x5, x6 = variables
    return -1 * (11125 * x1 + 16650 * x2 + 16500 * x3 + 11250 * x4 + 16950 * x5 + 21700 * x6)


def objective_municipality(variables):
    x1, x2, x3, x4, x5, x6 = variables
    return -1 * (x1 + x4)


cons = [{'type': 'ineq', 'fun': lambda x: -200 + x[0] + x[1] + x[2] + x[3] + x[4] + x[5]},
        {'type': 'ineq', 'fun': lambda x: -x[0] - x[1] - x[2] - x[3] - x[4] - x[5] + 260},

        {'type': 'ineq', 'fun': lambda x: -0.2 * sum(x) + x[0]},
        {'type': 'ineq', 'fun': lambda x: -x[0] + 0.3 * sum(x)},

        {'type': 'ineq', 'fun': lambda x: -0.15 * sum(x) + x[1]},
        {'type': 'ineq', 'fun': lambda x: -x[1] + 0.25 * sum(x)},

        {'type': 'ineq', 'fun': lambda x: -0.1 * sum(x) + x[2]},
        {'type': 'ineq', 'fun': lambda x: -x[2] + 0.15 * sum(x)},

        {'type': 'ineq', 'fun': lambda x: -0.15 * sum(x) + x[3]},
        {'type': 'ineq', 'fun': lambda x: -x[3] + 0.2 * sum(x)},

        {'type': 'ineq', 'fun': lambda x: -0.1 * sum(x) + x[4]},
        {'type': 'ineq', 'fun': lambda x: -x[4] + 0.2 * sum(x)},

        {'type': 'ineq', 'fun': lambda x: -0.05 * sum(x) + x[5]},
        {'type': 'ineq', 'fun': lambda x: -x[5] + 0.1 * sum(x)}
        ]

types = ['A', 'C', 'L', 'M', 'Q', 'S']

print('Optimizing on the project developer’s objective:')
result = minimize(objective_project_dev, np.array([1, 1, 1, 1, 1, 1]), method='SLSQP', constraints=cons)

print('The optimal solution is for:')
for i, n in enumerate(result.x):
    print(f'\t{round(n)} houses of type {types[i]}')

print(f'This will result in a profit of €{round(-1 * result.fun, 2)}')

print()
print('Optimizing on the municipality’s objective:')
result = minimize(objective_municipality, np.array([1, 1, 1, 1, 1, 1]), method='SLSQP', constraints=cons)

print(f'the maximum number of affordable houses is {round(-1 * result.fun)}')

print()
print('Optimizing using the constraint method:')
cons.append({'type': 'ineq', 'fun': lambda x: -130 + x[0] + x[3]})
result = minimize(objective_project_dev, np.array([1, 1, 1, 1, 1, 1]), method='SLSQP', constraints=cons)

print('The optimal solution is for:')
for i, n in enumerate(result.x):
    print(f'\t{round(n)} houses of type {types[i]}')

print(f'This will result in a profit of €{round(-1 * result.fun, 2)}')
