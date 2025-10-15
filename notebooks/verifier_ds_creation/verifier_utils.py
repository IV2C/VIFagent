import re
import numpy as np

PER_RANGE_SOLUTIONS = 1


def handle_def(template_code: str) -> list[str]:
    return [re.sub(r"§def\((.*?)\)", r"\1", template_code)]


def handle_range(template_code: str) -> list[str]:
    created_solution_code = [template_code]
    for f in reversed(
        list(re.finditer(r"§range\(([^,]+),([^,]+),([^)]+)\)", template_code))
    ):
        temp_solutions = []

        lower = float(f.group(1))
        higher = float(f.group(2))

        nb_range_sol = PER_RANGE_SOLUTIONS

        current_range = [i for i in np.linspace(lower, higher, nb_range_sol) if i != 0]
        for possible_range_float in current_range:
            for possible_solution in created_solution_code:
                create_code = (
                    possible_solution[: f.start()]
                    + str(possible_range_float)
                    + possible_solution[f.end() :]
                )
                temp_solutions.append(create_code)
        created_solution_code = temp_solutions
    return created_solution_code


def handle_rangei(template_code: str) -> list[str]:
    created_solution_code = [template_code]
    for f in reversed(list(re.finditer(r"§rangei\(([^,]+),([^)]+)\)", template_code))):
        temp_solutions = []

        lower = float(f.group(1)) - float(f.group(2))
        higher = float(f.group(1)) + float(f.group(2))

        nb_range_sol = PER_RANGE_SOLUTIONS

        current_range = [i for i in np.linspace(lower, higher, nb_range_sol) if i != 0]
        for possible_range_float in current_range:
            for possible_solution in created_solution_code:
                create_code = (
                    possible_solution[: f.start()]
                    + str(possible_range_float)
                    + possible_solution[f.end() :]
                )
                temp_solutions.append(create_code)
        created_solution_code = temp_solutions
    return created_solution_code


def handle_choices(template_code: str) -> list[str]:
    created_solution_code = [template_code]
    for f in reversed(
        list(re.finditer(r"§choice\((\[[^]]+\]),([^)]+)\)", template_code))
    ):
        temp_solutions = []

        choices: list = eval(f.group(1))

        for choice in choices:
            for possible_solution in created_solution_code:
                create_code = (
                    possible_solution[: f.start()]
                    + str(choice)
                    + possible_solution[f.end() :]
                )
                temp_solutions.append(create_code)
        created_solution_code = temp_solutions
    return created_solution_code
