"""
Python code for the shopping mall design problem example (Chapter 7.2)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

"""
In the first part of this script, the single objective optimization is shown. 
First, the three objectives and two constraints are defined as python functions. Next, the boundaries for the design 
variables x1 and x2 are defined, after which the optimizations can be done. The results of these optimizations are then 
printed as a Pandas DataFrame.

Note that objective 1 and 3 need to be maximized, whereas objective 2 should be minimized. Hence, only for objective 1 
and 3, the result is multiplied by -1.
"""


def objective_1(variables):
    """
    Objective to maximize the profit. Note that the returned value is multiplied by -1. This is since we use a
    minimization algorithm.

    :param variables: list with design variable values
    """
    x1 = variables[0]
    x2 = variables[1]
    return -1 * (160 * x1 + 80 * x2)


def objective_2(variables):
    """
    Objective to minimize the CO2 emissions.

    :param variables: list with design variable values
    """
    x1 = variables[0]
    x2 = variables[1]
    return 120 * x1 + 30 * x2


def objective_3(variables):
    """
    Objective to maximize the shopping potential. Note that the returned value is multiplied by -1. This is since we use
    a minimization algorithm.

    :param variables: list with design variable values
    """
    x1 = variables[0]
    x2 = variables[1]
    return -1 * (15 * x1 + 45 * x2)


def constraint_1(variables):
    """
    Constraint for optimization problem

    :param variables: list with design variable values
    """
    x1 = variables[0]
    x2 = variables[1]

    return 10000 - x1 - x2  # < 0


def constraint_2(variables):
    """
    Constraint for optimization problem

    :param variables: list with design variable values
    """
    x1 = variables[0]
    x2 = variables[1]

    return x1 + x2 - 3000  # < 0


# define a list in which all the constraints are combined.
cons = [{'type': 'ineq', 'fun': constraint_1},
        {'type': 'ineq', 'fun': constraint_2}]

# set bounds for all variables
b1 = (0, 5000)  # x1
b2 = (0, 7000)  # x2
bounds = (b1, b2)

# below, the three objectives are all optimized separately.
result_1 = minimize(fun=objective_1, x0=np.array([1, 1]), method='SLSQP', bounds=bounds, constraints=cons)
result_2 = minimize(fun=objective_2, x0=np.array([1, 1]), method='SLSQP', bounds=bounds, constraints=cons)
result_3 = minimize(fun=objective_3, x0=np.array([1, 1]), method='SLSQP', bounds=bounds, constraints=cons)

# We can add the results to a pandas DataFrame to make is prettier to print
values = np.zeros((3, 5))  # define empty numpy array to store the results in

x_1, x_2 = result_1.x  # results of optimization for just objective 1
values[0] = [x_1, x_2, objective_1(result_1.x), objective_2(result_1.x), objective_3(result_1.x)]

x_1, x_2 = result_2.x  # results of optimization for just objective 2
values[1] = [x_1, x_2, objective_1(result_2.x), objective_2(result_2.x), objective_3(result_2.x)]

x_1, x_2 = result_3.x  # results of optimization for just objective 3
values[2] = [x_1, x_2, objective_1(result_3.x), objective_2(result_3.x), objective_3(result_3.x)]

# add results to dataframe and print it
df = pd.DataFrame(np.round_(values), columns=['x1', 'x2', 'Profit', 'Emission', 'Potential'])
print(df)
print()

"""
The optimization problem can be expressed in preference functions.
The preference functions of this example are declared in the lines below.
"""

def objective_p1(variables):
    """
    Objective to maximize the profit preference. Note that the returned value is multiplied by -1. This is since we use
    a minimization algorithm.

    :param variables: list with design variable values
    """
    x1 = variables[0]
    x2 = variables[1]
    return -1 * (1 / 9600 * ((160 * x1 + 80 * x2) - 240000))


def objective_p2(variables):
    """
    Objective to maximize the CO2 emission preference. Note that the returned value is multiplied by -1. This is since
    we use a minimization algorithm.

    :param variables: list with design variable values
    """
    x1 = variables[0]
    x2 = variables[1]
    return -1 * (100 - (1 / 6600) * ((120 * x1 + 30 * x2) - 90000))


def objective_p3(variables):
    """
    Objective to maximize the shopping potential preference. Note that the returned value is multiplied by -1. This is
    since we use a minimization algorithm.

    :param variables: list with design variable values
    """
    x1 = variables[0]
    x2 = variables[1]
    return -1 * ((1 / 3150) * ((15 * x1 + 45 * x2) - 45000))


# the bounds and constraints are still the same, so we can now run the optimization for the three preference functions
result_p1 = minimize(objective_p1, np.array([1, 1]), method='SLSQP', bounds=bounds, constraints=cons)
result_p2 = minimize(objective_p2, np.array([1, 1]), method='SLSQP', bounds=bounds, constraints=cons)
result_p3 = minimize(objective_p3, np.array([1, 1]), method='SLSQP', bounds=bounds, constraints=cons)

# We can add the results to a pandas DataFrame to make is prettier to print
results = np.zeros((3, 5))  # define empty numpy array to store the results in

x_1, x_2 = result_p1.x  # results of optimization for just preference function 1
results[0] = [x_1, x_2, objective_p1(result_p1.x), objective_p2(result_p1.x), objective_p3(result_p1.x)]

x_1, x_2 = result_p2.x  # results of optimization for just preference function 2
results[1] = [x_1, x_2, objective_p1(result_p2.x), objective_p2(result_p2.x), objective_p3(result_p2.x)]

x_1, x_2 = result_p3.x  # results of optimization for just preference function 3
results[2] = [x_1, x_2, objective_p1(result_p3.x), objective_p2(result_p3.x), objective_p3(result_p3.x)]

# add results to dataframe and print it
df = pd.DataFrame(np.round_(results), columns=['x1', 'x2', 'Profit', 'Emission', 'Potential'])
print(df)

"""
Lastly, we can also plot the preference functions. This is done below.
"""

# create arrays for plotting continuous preference curves
c1 = np.linspace(0, 1200000)
c2 = np.linspace(0, 750000)
c3 = np.linspace(0, 360000)

# calculate the preference functions
p1 = 1 / 9600 * (c1 - 240000)
p2 = 100 - (1 / 6600) * (c2 - 90000)
p3 = (1 / 3150) * (c3 - 45000)

# create figure that plots all preference curves in subplots
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(c1, p1)
ax1.set_xlim((0, 1200000))
ax1.set_ylim((0, 100))
ax1.set_title('Profit')
ax1.set_xlabel('Profit [€]')
ax1.set_ylabel('Preference score')
ax1.grid()

fig = plt.figure()
ax2 = fig.add_subplot(1, 1, 1)
ax2.plot(c2, p2)
ax2.set_xlim((0, 750000))
ax2.set_ylim((0, 100))
ax2.set_title('CO2 Emission')
ax2.set_xlabel('Emissions [kg]')
ax2.set_ylabel('Preference score')
ax2.grid()

fig = plt.figure()
ax3 = fig.add_subplot(1, 1, 1)
ax3.plot(c3, p3)
ax3.set_xlim((0, 360000))
ax3.set_ylim((0, 100))
ax3.set_title('Shopping potential')
ax3.set_xlabel('Shopping potential [people]')
ax3.set_ylabel('Preference score')
ax3.grid()

plt.show()  # show figures
