import pathlib

from numpy import loadtxt
from numpy.testing import assert_allclose

from genetic_algorithm_pfm.tetra_pfm import TetraSolver

solver = TetraSolver()

HERE = pathlib.Path(__file__).parent


def test_compare_choicerobot():
    """
    Test to see if the python package TetraSolver is broken. The comparison is made with the solver on choicerobot.com
    itself. Trows AssertionError when the result is not the same, indicating the python code is broken.

    :return: None
    """

    data = loadtxt(f'{HERE}/data/test.csv', delimiter=';').tolist()
    desired_result = loadtxt(f'{HERE}/data/test_result.csv', delimiter=';')

    w = [0.25, 0.25, 0.25, 0.25]
    ret = solver.request(w=w, p=data)

    print("{:<16} {:<16}".format('Desired result', 'Result Tetra'))
    for i in range(len(ret)):
        print("{:<16} {:<16}".format(round(desired_result[i], 4), round(ret[i], 4)))

    assert_allclose(actual=ret, desired=desired_result, rtol=1e-5, atol=5e-5, err_msg='Returned data is wrong')
    print('Test was successful')
    return


if __name__ == '__main__':
    test_compare_choicerobot()
