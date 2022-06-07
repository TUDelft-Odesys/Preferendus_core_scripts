"""Example from Preference-based optimization reader. Credits to Dmitry Zhilyaev for creating the example

Code adapted by Harold van Heukelum
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tetra_pfm import TetraSolver
from scipy.optimize import minimize

# create arrays for plotting continuous preference curves
c1 = np.linspace(0, 1200000)
c2 = np.linspace(0, 750000)
c3 = np.linspace(0, 360000)

p1 = 1 / 9600 * (c1 - 240000)
p2 = 100 - (1 / 6600) * (c2 - 90000)
p3 = (1 / 3150) * (c3 - 45000)

# create figure that plots all preference curves and the preference scores of the returned results of the GA
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Preference functions')

ax1.plot(c1, p1)
ax1.set_xlim((0, 1200000))
ax1.set_ylim((0, 100))
ax1.set_title('Profit')
ax1.set_xlabel('Profit [€]')
ax1.set_ylabel('Preference score')
ax1.grid()

ax2.plot(c2, p2)
ax2.set_xlim((0, 750000))
ax2.set_ylim((0, 100))
ax2.set_title('CO2 Emission')
ax2.set_xlabel('Emissions [kg]')
ax2.set_ylabel('Preference score')
ax2.grid()

ax3.plot(c3, p3)
ax3.set_xlim((0, 360000))
ax3.set_ylim((0, 100))
ax3.set_title('Shopping potential')
ax3.set_xlabel('Shopping potential [people]')
ax3.set_ylabel('Preference score')
ax3.grid()

# import Tetra solver
solver = TetraSolver()

# set weights
w1 = 1 / 3
w2 = 1 / 3
w3 = 1 / 3


def objective_p1(variables):
    x1 = variables[0]
    x2 = variables[1]

    return -1 * (1 / 9600 * ((160 * x1 + 80 * x2) - 240000))


def objective_p2(variables):
    x1 = variables[0]
    x2 = variables[1]

    return -1 * (100 - (1 / 6600) * (120 * x1 + 30 * x2 - 90000))


def objective_p3(variables):
    x1 = variables[0]
    x2 = variables[1]

    return -1 * ((1 / 3150) * (15 * x1 + 45 * x2 - 45000))


# set bounds for all variables
b1 = (0, 5000)  # x1
b2 = (0, 7000)  # x2
bounds = (b1, b2)


def constraint_1(variables):
    x1 = variables[0]
    x2 = variables[1]

    return 10000 - x1 - x2  # < 0


def constraint_2(variables):
    x1 = variables[0]
    x2 = variables[1]

    return x1 + x2 - 3000  # < 0


# define list with constraints
cons = [{'type': 'ineq', 'fun': constraint_1},
        {'type': 'ineq', 'fun': constraint_2}]

result_p1 = minimize(objective_p1, np.array([1, 1]), method='SLSQP', bounds=bounds, constraints=cons)

result_p2 = minimize(objective_p2, np.array([1, 1]), method='SLSQP', bounds=bounds, constraints=cons)

result_p3 = minimize(objective_p3, np.array([1, 1]), method='SLSQP', bounds=bounds, constraints=cons)

print(result_p3.x)
results = np.zeros((3, 3))
results[0] = [objective_p1(result_p1.x), objective_p2(result_p1.x), objective_p3(result_p1.x)]
results[1] = [objective_p1(result_p2.x), objective_p2(result_p2.x), objective_p3(result_p2.x)]
results[2] = [objective_p1(result_p3.x), objective_p2(result_p3.x), objective_p3(result_p3.x)]

df = pd.DataFrame(np.round_(results, 2), columns=['Criterion 1', 'Criterion 2', 'Criterion 3'])
print(df)

plt.show()
