# API connection to Tetra

This package repository the module to send to and receive data from the Tetra API. It will build the XMl for you and
returns the data as a list. Please note that credentials are needed for the API and these credentials are not free 
to share!

# How to use

Aggregate the preference scores by using the class TetraSolver (`from tetra_pfm import TetraSolver`). It takes two lists
as inputs:

- argument 1: list with all weights.
- argument 2: list with all preference scores (p1, p2, etc.).

TetraSolver returns one list with the preference scores for all alternatives (= members of the population of the GA).

# Contact

For questions, suggestions, or remarks, you can contact me by harold.van.heukelum@boskalis.com.

# Closing remarks

This repository is licensed under the [MIT licence](https://choosealicense.com/licenses/mit/).
See [also here](https://github.com/HaroldPy11/PFM_core_scripts/blob/main/LICENSE).

Copyright (c) 2022 Harold van Heukelum

