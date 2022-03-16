# Genetic Algorithm (GA) for Preference Function Modelling

This repository contains the optimization module for Preference Function Modelling. A Genetic Algorithm is used for the
optimization.

*Sources:*

- *Brownlee, J. (2021, March 3). Simple genetic algorithm from scratch in Python. Machine Learning Mastery. Retrieved
  November 25, 2021, from https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/*
- *Kramer, O. (2008). Self-adaptive heuristics for evolutionary computation. Springer.*
- *Coello, C. A. C. (2002). Theoretical and numerical constraint-handling techniques used with evolutionary algorithms:
  a survey of the state of the art. Computer methods in applied mechanics and engineering, 191(11-12), 1245-1287.*
- *Morales, A. K., & Quezada, C. V. (1998, September). A universal eclectic genetic algorithm for constrained
  optimization. In Proceedings of the 6th European congress on intelligent techniques and soft computing
  (Vol. 1, pp. 518-522).*

# How to use

Import the GA by `from genetic_algorithm_pfm import genetic_algorithm`. It takes the following arguments:

- **objective:** you objective function (see previous section).
- **constraints:** list with all constraints, in format: [[type, function], etc.]. Type can either be 'eq' for equality
  constraint or 'ineq' of inequality constraint.
- **bounds:** list with bounds of variabels, in format [[lb var, ub var1],[lb var2, ub var2], etc.].
- **cons_handler:** Two types of constraint handler can be chosen: simple penalty function (default), or Coello
  Non-Dominance handler. The latter is better for highly constraint problems. See source code for the papers in which
  the handlers are described. NB. handlers are only limited tested for equality constraints.
- **options:** dictionary that contain the parameters of the GA:
    - n_bits: number of bits in bit string (default: 24);
    - n_iter: maximal number of iterations (default: 400);
    - n_pop: population size (default: 500);
    - r_cross: cross-over rate (default: 0.85);
    - max_stall: maximal generation with no improvements before the GA stops (default: 15);
    - tol: improvement smaller than this value is neglected by the GA (default: 1e-15)
- **verbose:** print progress of GA to console. True by default.
- **tetra:** Should be true when using the GA in combination with Tetra, to allow for correct assessment. True by
  default.

# Contact

For questions, suggestions, or remarks, you can contact me by harold.van.heukelum@boskalis.com.

# Closing remarks

This repository is licensed under the [MIT licence](https://choosealicense.com/licenses/mit/).
See [also here](https://github.com/HaroldPy11/PFM_core_scripts/blob/main/LICENSE).

Copyright (c) 2022 Harold van Heukelum

