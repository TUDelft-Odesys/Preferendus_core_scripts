# Core Scripts for multi-objective, _a priori_ optimization by means of Tetra

This repository contains the scripts needed to solve a multi-objective problem _a priori_ with Tetra. For this, you need
the two 'packages' genetic_algorithm_pfm and tetra_pfm. The examples of the reader are listed in the main directory of 
the repository.

# How to use

## Download the code

You can download this repository:

- Via the terminal: `git clone https://github.com/TUDelft-Odesys/Preferendus_core_scripts.git`
- By downloading the zip-file via the button 'Code' --> Download ZIP

## Objective function

The objective function takes an array (type=ndarray) as input. This array has shape n-by-m, with n the population size
of the GA and m the number of variables. Extract the variables as n-by-1 arrays by using subscripts ([:,0], [:,1], etc.)

Aggregate the preference scores by using the class TetraSolver (`from tetra_pfm import TetraSolver`). It takes two lists
as inputs:

- argument 1: list with all weights.
- argument 2: list with all preference scores (p1, p2, etc.).

TetraSolver returns one list with the preference scores for all the alternatives (= members of the population of the GA).

## Genetic Algorithm (GA)

Import the GA by `from genetic_algorithm_pfm import GenetricAlgorithm`. First call GeneticAlgorithm by using `ga = 
GeneticAlgorithm(*args)` in your script. It takes the following arguments (`*args`):

- **objective:** you objective function (see previous section).
- **constraints:** list with all constraints, in format: [[type, function], etc.]. Type can either be 'eq' for equality
  constraint or 'ineq' of inequality constraint.
- **bounds:** list with bounds of variabels, in format [[lb var, ub var1],[lb var2, ub var2], etc.].
- **cons_handler:** Two types of constraint handler can be chosen: simple penalty function (default), or Coello
  Non-Dominance handler. The latter is better for highly constraint problems. See source code for the papers in which
  the handlers are described. NB. handlers are only limited tested for equality constraints!
- **options:** dictionary that contain the parameters of the GA:
    - n_bits: number of bits in bit string;
    - n_iter: maximal number of iterations;
    - n_pop: population size;
    - r_cross: cross-over rate;
    - max_stall: maximal generation with no improvements before the GA stops;
    - tol: improvement smaller than this value is neglected by the GA ;
    - tetra: should be true when using the GA in combination with Tetra, to allow for correct assessment.
    - var_type: type of variable ('real', 'int', 'bool'). Leave undefined when you have a mixed-integer problem! Default type is 'real'.
    - var_type_mixed: list with the types of your variables. Length is equal to the number of variables. Type can either be 'real', 'int' or 'bool'.

To run the GA, use `ga.run()`.

## Assessment of GA in combination with Tetra

Since the latest version of Tetra, all results are relative to the input and always sorted from 0 to 100. This makes the
problem not deterministic and makes it not possible to use the normal workflow of a GA. To account for this, an
additional step is added to the GA:

1. for every generation n; the best result from generation n is combined with the best results from all previous
   generations (n-1,n-2,...,n=0) in one array.
2. This array is put through Tetra separately and this results in one array with scores from Tetra.
3. If the score for the set of variables of generation n = 100 (ie. the best of all generations), the GA takes this as
   the new best score.
4. The stall counter is determined by checking how many generations have a score of 100.

# Contact

For questions, suggestions, or remarks, you can contact me by harold.van.heukelum@boskalis.com.

# Closing remarks

This repository is currently licensed under the [GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/).
See [also here](https://github.com/TUDelft-Odesys/Preferendus_core_scripts/blob/main/LICENSE).

Copyright (c) 2022 Harold van Heukelum
