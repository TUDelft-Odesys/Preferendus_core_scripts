# Core Scripts for multi-objective, _a priori_ optimization by means of Tetra

This repository contains the scripts needed to solve a multi-objective problem _a priori_ with Tetra. For this, you need
the two 'packages' genetic_algorithm_pfm and tetra_pfm. Additionally, an example project is added in /Example/.

# How to use

## Download the code

You can download this repository:

- Via the terminal: git clone https://github.com/HaroldPy11/PFM_core_scripts.git;
- By downloading the zip-file via Code --> Download ZIP.

## Objective function

The objective function takes an array (type=ndarray) as input. This array has shape n-by-m, with n the population size
of the GA and m the number of variables. Extract the variables as n-by-1 arrays by using subscripts ([:,0], [:,1], etc.)

Aggregate the preference scores by using the class TetraSolver (`from tetra_pfm import TetraSolver`). It takes two lists
as inputs:

- argument 1: list with all weights.
- argument 2: list with all preference scores (p1, p2, etc.).

TetraSolver returns one list with the preference scores for all alternatives (= members of the population of the GA).

## Genetic Algorithm (GA)

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

## Assessment og GA in combination with Tetra

Since the latest version of Tetra, all results are relative to the input and always sorted from 0 to 100. This makes the
problem not deterministic and makes it not possible to use the normal workflow of a GA. To account for this, an
additional step is added to the GA:

1. for every generation n; the best result from generation n is combined with the best results from all previous
   generations (n-1,n-2,...,n=0) in one array.
2. This array is put through Tetra separately and this results in one array with scores from Tetra.
3. If the score for the set of variables of generation n = 100 (ie. the best of all generations), the GA takes this as
   the new best score.
4. The stall counter is determined by checking how many generations have passed since the last 100-score. If stall
   counter == max_stall, the GA stops and returns the results from the generation which scored 100.

# Contact

For questions, suggestions, or remarks, you can contact me by harold.van.heukelum@boskalis.com.

# Closing remarks

This repository is currently licensed under the [MIT licence](https://choosealicense.com/licenses/mit/).
See [also here](https://github.com/HaroldPy11/PFM_core_scripts/blob/main/LICENSE).

Copyright (c) 2022 Harold van Heukelum
