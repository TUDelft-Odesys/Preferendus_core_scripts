"""
This algorithm is based on the sources listed below. Code is adapted for PFM by Harold van Heukelum.

Sources:
Brownlee, J. (2021, March 3). Simple genetic algorithm from scratch in Python. Machine Learning Mastery. Retrieved
November 25, 2021, from https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/.

Kramer, O. (2008). Self-adaptive heuristics for evolutionary computation. Springer.

Copyright (c) Harold van Heukelum, 2021
"""
from time import perf_counter

from numpy import array, mean, where, unique, max, round_, count_nonzero
from numpy.random import randint

from ._constraints import _const_handler
from ._decoder import _decode
from ._nextgen import _selection, _mutation, _crossover


class Colors:
    OK = '\033[92m'  # GREEN
    WARNING = '\033[93m'  # YELLOW
    FAIL = '\033[91m'  # RED
    RESET = '\033[0m'  # RESET COLOR


def genetic_algorithm(objective, constraints: list, bounds: list, cons_handler: str = 'simple', options: dict = None,
                      verbose=True, tetra=True):
    """
    Optimization algorithm based on survival of the fittest. Searches for the minimal value.

    The following parameters can be defined in the option dictionary:
    n_bits: number of bits per variable. See docstring of decode function for details on specification of number
    of bits
    n_iter: maximum number of generations before algorithm is stopped. Prevents infinite loop
    n_pop: size of the population that is generated per generation, ie. number of guesses per generation
    r_cross: value between 0 and 1 that determine how often crossover is performed
    max_stall: number of generations that must have a change no more than the tolerance value before the
    algorithm is stopped
    tol: tolerance for change. If improvement is below this value, it is seen as the same value.

    :param objective: the function to minimize. Must be in the form f(x, *args) with x is a 2-D array of width
    len(bounds) and length n_pop
    :param constraints: list if constraint-functions
    :param bounds: boundaries for variables in x. Every variable in x should have a boundary!
    :param cons_handler: simple (default) or CND (Coello non-dominance)
    :param options: dictionary that contains all parameters for the GA. See doc string for explanation of these
    parameters
    :param verbose: Print progress to console
    :param tetra: when Tetra is used, special measures need to be taken to deal with the relativity involved
    :return: the optimal score of the objective function, and the scores for the variables
    """

    # gather user specified parameters or fall back to default values
    if options is None:
        options = {}
    n_bits: int = options.get('n_bits', 24)
    n_iter: int = options.get('n_iter', 400)
    n_pop: int = options.get('n_pop', 500)
    r_cross: float = options.get('r_cross', 0.85)
    max_stall: int = options.get('max_stall', 15)
    tol: float = options.get('tol', 1e-15)

    # set mutation rate based on parameters and check if population is even to allow for elitism
    r_mut = 1 / (float(n_bits) * len(bounds))
    assert n_pop % 2 == 0, 'N_pop must be even'

    # initial population of random bitstrings. Size is len(bounds)*n_bit-by-n_pop
    pop = [randint(0, 2, n_bits * len(bounds)).tolist() for _ in range(n_pop)]

    # set initial best and best_eval
    best, best_eval = randint(0, 2, n_bits * len(bounds)).tolist(), 1e6

    stall_counter = 0
    gen = 0
    tic = perf_counter()  # start timer
    check_array_complete = list()

    # print headers for console output
    if verbose:
        print(
            "{:<12} {:<12} {:<16} {:<12} {:<12} {:<12}".format('Generation', 'Best score', 'Mean', 'Max stall',
                                                               'Diversity',
                                                               'Number of non-feasible results'))

    # loop through generations
    for gen in range(n_iter):
        best_eval_old = best_eval

        # decode population. Should be np.array to make masks possible
        decoded = array([_decode(bounds, n_bits, p) for p in pop])

        # check diversity:
        check_div = round(max(unique(decoded[:, 0], return_counts=True)[1]) / len(pop), 3)

        # evaluate all candidates in the population
        scores = objective(decoded)
        scores_feasible, length_cons = _const_handler(cons_handler, constraints, decoded, scores)

        def print_status():
            """Print intermittent results to console when verbose is True"""
            if verbose:
                print("{:<12} {:<12} {:<16} {:<12} {:<12} {:<12}".format(gen, round(best_eval, 4),
                                                                         round(float(mean(scores_feasible)), 4),
                                                                         stall_counter, check_div, length_cons))

        # check for new best solution; print current bests and stall counter to console
        if tetra:
            # add best result of current run to check_array
            check_array_complete.append(decoded[where(array(scores_feasible) ==
                                                      min(scores_feasible))[0][0]].tolist())

            # evaluate check_array through Tetra (via objective function)
            result = objective(array(check_array_complete))

            # check for improvement
            if result[-1] <= min(result):
                best_eval = min(scores_feasible)
                best = pop[where(array(scores_feasible) == min(scores_feasible))[0][0]]

            # set stall counter
            result = array(round_(result, 5))
            if -100.0 in result:
                counter_1 = len(result) - max(where(result == -100.))
                if result[-1] == -100.:
                    counter_2 = count_nonzero(result == -100.)
                else:
                    counter_2 = 1
                stall_counter = max([counter_1, counter_2])
            else:
                stall_counter = 1

            """
            The code below is an second method to deal with relative Tetra scores. It is less elegant and 
            computational expensive, but be free to use it.

            decode_old_best = _decode(bounds, n_bits, best_old)
            decode_current_best = decoded[where(array(scores_feasible) == min(scores_feasible))[0][0]]

            if allclose(decode_old_best, decode_current_best, atol=1e-4, rtol=1e-10) is False:
                stall_counter = 0
                best_eval = min(scores_feasible)
                best = pop[where(array(scores_feasible) == min(scores_feasible))[0][0]]
                plot_array.append(decoded[where(array(scores_feasible) == min(scores_feasible))[0][0]])
            else:
                stall_counter += 1
            """

        else:  # normal GA evaluation
            mask = array(scores_feasible) < best_eval
            if mask.any():
                stall_counter = 0
                best_eval = min(scores_feasible)
                best = pop[where(array(scores_feasible) == min(scores_feasible))[0][0]]

            if abs(best_eval_old - best_eval) < tol:
                stall_counter += 1

        print_status()
        if stall_counter >= max_stall:
            if verbose:
                print(f'Stopped at gen {gen}')
            break

        # select parents for next generation
        selected = [_selection(pop, scores_feasible) for _ in range(n_pop - 2)]

        # create the next generation
        children = [best, best]  # elitism
        i = -1  # to allow for IndexError handling
        try:
            for i in range(0, len(selected), 2):
                # get selected parents in pairs
                p1, p2 = selected[i], selected[i + 1]

                # crossover and mutation
                for c in _crossover(p1, p2, r_cross):
                    # mutation
                    _mutation(c, r_mut)

                    # store for next generation
                    children.append(c)
        except IndexError as err:
            print(i)
            raise err

        # replace population
        pop = children
        assert len(pop) == n_pop, f'Pop array is not equal after children are made. ' \
                                  f'It is now {len(pop)} and should be n_pop = {n_pop}'

    decoded = _decode(bounds, n_bits, best)  # get final values of variables based on best generation
    toc = perf_counter()

    if verbose:
        print(f'Execution time was {toc - tic:0.4f} seconds')

    # some fail safes
    if gen == n_iter - 1:
        print(Colors.FAIL + 'The iteration is stopped since the max number is reached. The results might be '
                            'incorrect! Please be cautious.' + Colors.RESET)
    elif gen < max_stall + 5:
        print(Colors.FAIL + 'The number of generations is terribly close to the number of max stall iterations. '
                            'This suggests a too fast convergence and wrong results.')
        print('Please be careful in using these results and assume they are wrong unless proven otherwise!'
              + Colors.RESET)

    return best_eval, decoded
