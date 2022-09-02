"""
Python code for the multi-stakeholder urban design problem example
"""
import numpy as np
from scipy.optimize import minimize

"""
In this example we will use the scipy.optimize.minimize function to search for the optimal solution for the objectives.
Since this is a rather simple, linear problem, we can use a simple and fast solver like the one provided with scipy.

To be able to optimize, we need to define the objectives as an python function. Similar we need to define a function for 
the constraints. Since we have so many constraints, a more compact approached is used in this example. More on that 
later on.

For documentation on scipy.optimize.minimize please see the website of scipy or use the help() function.
"""


def objective_project_dev(variables):
    """
    Objective function to maximize the profit of the project developer. Note that the returned value is multiplied by
    -1. This is since we use a minimization algorithm.

    :param variables: list with design variable values
    """
    x1, x2, x3, x4, x5, x6 = variables
    return -1 * (11125 * x1 + 16650 * x2 + 16500 * x3 + 11250 * x4 + 16950 * x5 + 21700 * x6)


def objective_municipality(variables):
    """
    Objective function to maximize the number of affordable houses. Note that the returned value is multiplied by
    -1. This is since we use a minimization algorithm.

    :param variables: list with design variable values
    """
    x1, x2, x3, x4, x5, x6 = variables
    return -1 * (x1 + x4)


"""
In previous examples, the constraints were all programmed as seperate functions. Since we have so much functions here, 
this approach would mean we need a lot of space. This would result in a file that is hard to read and thus hard to fix 
when errors occur.

Instead, the lambda function is used. The lambda function makes it possible to define a function on one line of code. It
functions the same as a normal python function, however, requires a lot less space. It would require to much effort to 
go into detail on the functionality here, especially since there are good tutorials about. The interested reader is hence
referred to https://realpython.com/python-lambda/ 

The short version is that if we look to line 54, it is basically the same as:

(1)     def constraint_1(x):
            return -200 + x[0] + x[1] + x[2] + x[3] + x[4] + x[5]
    
(2)     {'type': 'ineq', 'fun': constraint_1}
"""

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

types = ['A', 'C', 'L', 'M', 'Q', 'S']  # make a list that contains the names of all types of houses

# below, the project developer objective is optimized and the result is printed.
print('Optimizing on the project developer’s objective:')
result = minimize(objective_project_dev, np.array([1, 1, 1, 1, 1, 1]), method='SLSQP', constraints=cons)

print('The optimal solution is for:')
for i, n in enumerate(result.x):
    print(f'\t{round(n)} houses of type {types[i]}')

print(f'This will result in a profit of €{round(-1 * result.fun, 2)}')

# secondly, the municipality's objective is optimized
print()
print('Optimizing on the municipality’s objective:')
result = minimize(objective_municipality, np.array([1, 1, 1, 1, 1, 1]), method='SLSQP', constraints=cons)

print('The optimal solution is for:')
for i, n in enumerate(result.x):
    print(f'\t{round(n)} houses of type {types[i]}')
    # the observant student will note that the sum of the printed number of houses is more than 260. This is due to a
    # rounding error. This can be mitigated by using an optimization algorithm that is fit for integer problems.

print(f'the maximum number of affordable houses is {round(-1 * result.fun)}')

"""
Lastly, the objective for the project developer is optimized again, only now the municipality's objective is added as 
a constraint. This constraint is added to the list with constraints declared on line 59 by using the append method.
"""

print()
print('Optimizing using the constraint method:')
cons.append({'type': 'ineq', 'fun': lambda x: -130 + x[0] + x[3]})
result = minimize(objective_project_dev, np.array([1, 1, 1, 1, 1, 1]), method='SLSQP', constraints=cons)

print('The optimal solution is for:')
for i, n in enumerate(result.x):
    print(f'\t{round(n)} houses of type {types[i]}')

print(f'This will result in a profit of €{round(-1 * result.fun, 2)}')
