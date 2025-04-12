import importlib


def get_problem(problem_number):
    """Returns an istance of the problem class for a given problem number.

    Keyword arguments:
        problem_number  -- number of the BVP (between 19 and 33)
    """
    str_number = str(problem_number)
    if (problem_number < 10):
        str_number = "0" + str_number
    curr_name = "apps.bvp_" + str_number
    try:
        curr_module = importlib.import_module(curr_name)
    except ModuleNotFoundError:
        raise ValueError("This problem does not exist.")
    return curr_module.Problem()
