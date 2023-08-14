# Core scripts for multi-objective, _a priori_ optimization via the Preferendus

This repository contains the core elements of the Preferendus, including all examples and exemplars described in the
book Open Design Systems (to be published in 2023 by IOS Press). For the details about the Preferences, the reader is referred to 
*Van Heukelum, H. J., Binnekamp, R., Wolfert, A.R.M. 2022. Human Preference and Asset Performance Systems Design Integration
[Manuscript Submitted for Publication]*.

Note: most code is already annotated, however, improvements will be made in the near future

# Download the code

You can download this repository:

- Via the terminal: git clone https://github.com/TUDelft-Odesys/Preferendus_core_scripts.git;
- By downloading the zip-file via the button *Code* --> *Download ZIP*.

# Genetic Algorithm (GA)

Import the GA by `from genetic_algorithm_pfm import GenetricAlgorithm`. First call GeneticAlgorithm by using `ga = 
GeneticAlgorithm(*args)` in your script. It takes the following arguments:

- **objective:** you objective function.
- **constraints:** list with all constraints, in format: [[type, function], etc.]. Type can either be 'eq' for equality
  constraint or 'ineq' of inequality constraint.
- **bounds:** list with bounds of variables, in format [[lb var, ub var1],[lb var2, ub var2], etc.].
- **cons_handler:** Two types of constraint handler can be chosen: simple penalty function (default), or Coello
  Non-Dominance handler. The latter is better for highly constraint problems. See source code for the papers in which
  the handlers are described. NB. handlers are only tested for inequality constraints!
- **options:** dictionary that contain the parameters of the GA:
    - n_bits: number of bits in bit string (default: 24);
    - n_iter: maximal number of iterations (default: 400);
    - n_pop: population size (default: 500);
    - r_cross: cross-over rate (default: 0.80);
    - max_stall: maximal generation with no improvements before the GA stops (default: 15);
    - tol: improvement smaller than this value is neglected by the GA (default: 1e-15);
    - var_type: type of variable ('real', 'int', 'bool'). Default type is 'real'. **Leave undefined when you have a mixed-integer problem!**
    - var_type_mixed: list with the types of your variables. **Length should be equal to the number of variables.** Type can either be 'real', 'int' or 'bool';
    - aggregation: the GA can be used as an ordinary GA by setting the aggregation to None (default). To call the Preferendus, set this value to either 'tetra' or 'minmax' (depending on the method you want to use);
    - mutation_rate_order: The mutation can be influenced via this parameter (default: 2);
    - elitism percentage: this parameter influences the percentage of the population that is considered in the elitism (default: 15%) 

To run the GA, use `ga.run()`.

# Contact

For questions, suggestions, or remarks, you can contact me by support@odesys.nl.

# Closing remarks

This repository is currently licensed under the [MIT licence](https://choosealicense.com/licenses/mit/).

Copyright (c) 2023 Harold van Heukelum
