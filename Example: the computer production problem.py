"""
Python code for the computer production problem example (Chapter 5.2 Example 1)
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
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
eq = -1 * np.array([300, 500])

# next, define the constraints. For this we first provide a matrix A with all the coefficients x1 and x2 are multiplied.
A = np.array([[1, 0], [0, 1], [1, 2]])

# next we determine the upper bounds as vectors
b_u = np.array([60, 50, 120])

# finally, we need to define the lower bound. In our case, these are taken as 0
b_l = np.full_like(b_u, 0)

# we can now define the LinearConstraint
constraints = LinearConstraint(A, b_l, b_u)

# the integrality array will tell the algorithm it should take the variables as integers.
integrality = np.ones_like(eq)

"""Run the optimization"""

result = milp(c=eq, constraints=constraints, integrality=integrality)

print(f'The optimal solution is for producing {result.x[0]} basic computers and '
      f'{result.x[1]} advanced computers.')

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
