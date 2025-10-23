import math
import random
import re
import numpy as np

from notebooks.verifier_ds_creation.invalid_variant_creation import (
    BW_COLORS,
    XCOLOR_COLOR_LIST,
    XCOLOR_SHADE_LIST,
    try_generate_incorect_code,
)
from vif.utils.image_utils import nmse
from vif.utils.renderer.tex_renderer import TexRenderer

random.seed(100)
PER_RANGE_SOLUTIONS = 2
MAX_REG_AMOUNT = 5
isint = lambda v: math.floor(v) == v


def handle_def(template_code: str) -> list[str]:
    return [re.sub(r"§def\((.*?)\)", r"\1", template_code)]


def correct_from_range(template_code: str) -> list[str]:
    created_solution_code = [template_code]
    found_ranges = list(
        re.finditer(r"§range\(([^,]+),([^,]+),([^)]+)\)", template_code)
    )

    if len(found_ranges) > MAX_REG_AMOUNT:
        found_ranges = random.sample(found_ranges, MAX_REG_AMOUNT)
    for f in reversed(found_ranges):
        temp_solutions = []

        lower = float(f.group(1))
        higher = float(f.group(2))

        ranges_are_ints = all(isint(value) for value in [lower, higher])

        nb_range_sol = PER_RANGE_SOLUTIONS
        current_range = [i for i in np.linspace(lower, higher, nb_range_sol)]

        for possible_range_float in current_range:
            for possible_solution in created_solution_code:
                create_code = (
                    possible_solution[: f.start()]
                    + str(
                        possible_range_float
                        if not ranges_are_ints
                        else int(possible_range_float)
                    )
                    + possible_solution[f.end() :]
                )
                temp_solutions.append(create_code)
        created_solution_code = temp_solutions
    return created_solution_code


def correct_from_rangei(template_code: str) -> list[str]:
    created_solution_code = [template_code]
    found_ranges = list(re.finditer(r"§rangei\(([^,]+),([^)]+)\)", template_code))

    if len(found_ranges) > MAX_REG_AMOUNT:
        found_ranges = random.sample(found_ranges, MAX_REG_AMOUNT)

    for f in reversed(found_ranges):
        temp_solutions = []

        lower = float(f.group(1)) - float(f.group(2))
        higher = float(f.group(1)) + float(f.group(2))

        ranges_are_ints = all(
            isint(value) for value in [float(f.group(2)), float(f.group(1))]
        )

        nb_range_sol = PER_RANGE_SOLUTIONS

        current_range = [i for i in np.linspace(lower, higher, nb_range_sol)]
        for possible_range_float in current_range:
            for possible_solution in created_solution_code:
                create_code = (
                    possible_solution[: f.start()]
                    + str(
                        possible_range_float
                        if not ranges_are_ints
                        else int(possible_range_float)
                    )
                    + possible_solution[f.end() :]
                )
                temp_solutions.append(create_code)
        created_solution_code = temp_solutions
    return created_solution_code


def correct_from_choices(template_code: str) -> list[str]:
    created_solution_code = [template_code]
    found_ranges = list(re.finditer(r"§choice\((\[[^]]+\]),([^)]+)\)", template_code))
    if len(found_ranges) > MAX_REG_AMOUNT:
        found_ranges = random.sample(found_ranges, MAX_REG_AMOUNT)

    for f in reversed(found_ranges):
        temp_solutions = []

        choices: list = eval(f.group(1))

        choices_are_ints = all(
            str(value).isnumeric() and isint(value) for value in choices
        )

        for choice in choices:
            for possible_solution in created_solution_code:
                create_code = (
                    possible_solution[: f.start()]
                    + (str(choice) if not choices_are_ints else str(int(choice)))
                    + possible_solution[f.end() :]
                )
                temp_solutions.append(create_code)
        created_solution_code = temp_solutions
    return created_solution_code


INCORRECT_DISTANCE_RATIO = (
    3 / 10
)  # 3 tenth of the range of possible solution away from the lower and higher limits


def incorrect_from_range(template_code: str) -> list[str]:
    """Generates len(number of ranges)*2 invalid solutions from ranges in template code

    Args:
        template_code (str): The template code

    """

    found_ranges = list(
        re.finditer(r"§range\(([^,]+),([^,]+),([^)]+)\)", template_code)
    )
    if len(found_ranges) == 0:
        return []
    if len(found_ranges) > MAX_REG_AMOUNT:
        found_ranges = random.sample(found_ranges, MAX_REG_AMOUNT)
    identity = np.identity(len(found_ranges))
    created_solution_code = [
        [template_code for _ in range(2)] for _ in range(len(found_ranges))
    ]
    # *2 for lower than lower range and higher than higher range
    for id, f in enumerate(reversed(found_ranges)):

        lower = float(f.group(1))
        higher = float(f.group(2))
        default = float(f.group(3))

        for sol_id, is_incorrect in enumerate(identity[id]):
            if is_incorrect:
                current_distance = INCORRECT_DISTANCE_RATIO * (higher - lower)
                value_used = [(lower - current_distance), (higher + current_distance)]
            else:
                value_used = [default, default]

            sol1 = created_solution_code[sol_id][0]
            sol2 = created_solution_code[sol_id][1]
            created_solution_code[sol_id][0] = (
                sol1[: f.start()] + str(value_used[0]) + sol1[f.end() :]
            )
            created_solution_code[sol_id][1] = (
                sol2[: f.start()] + str(value_used[1]) + sol2[f.end() :]
            )

    return [code for codes in created_solution_code for code in codes]


def incorrect_from_rangei(template_code: str) -> list[str]:

    found_ranges = list(re.finditer(r"§rangei\(([^,]+),([^)]+)\)", template_code))
    if len(found_ranges) == 0:
        return []
    if len(found_ranges) > MAX_REG_AMOUNT:
        found_ranges = random.sample(found_ranges, MAX_REG_AMOUNT)
    identity = np.identity(len(found_ranges))
    created_solution_code = [
        [template_code for _ in range(2)] for _ in range(len(found_ranges))
    ]
    # *2 for lower than lower range and higher than higher range
    for id, f in enumerate(reversed(found_ranges)):

        default = float(f.group(1))
        higher = default + float(f.group(2))
        lower = default - float(f.group(2))

        for sol_id, is_incorrect in enumerate(identity[id]):
            if is_incorrect:
                current_distance = INCORRECT_DISTANCE_RATIO * (higher - lower)
                value_used = [(lower - current_distance), (higher + current_distance)]
            else:
                value_used = [default, default]

            sol1 = created_solution_code[sol_id][0]
            sol2 = created_solution_code[sol_id][1]
            created_solution_code[sol_id][0] = (
                sol1[: f.start()] + str(value_used[0]) + sol1[f.end() :]
            )
            created_solution_code[sol_id][1] = (
                sol2[: f.start()] + str(value_used[1]) + sol2[f.end() :]
            )

    return [code for codes in created_solution_code for code in codes]


def incorrect_from_choices(template_code: str) -> list[str]:

    found_ranges = list(re.finditer(r"§choice\((\[[^]]+\]),([^)]+)\)", template_code))
    if len(found_ranges) == 0:
        return []
    if len(found_ranges) > MAX_REG_AMOUNT:
        found_ranges = random.sample(found_ranges, MAX_REG_AMOUNT)

    identity = np.identity(len(found_ranges))
    created_solution_code = [template_code for _ in range(len(found_ranges))]
    # *2 for lower than lower range and higher than higher range
    for id, f in enumerate(reversed(found_ranges)):
        choices: list = [str(v) for v in eval(f.group(1))]
        default = f.group(2)

        for sol_id, is_incorrect in enumerate(identity[id]):

            if created_solution_code[sol_id] == None:
                continue

            if is_incorrect:
                value_used = guess_invalid_choice(choices)
                if value_used == None:  # Could not guess an invalid choice
                    created_solution_code[sol_id] = None
                    continue
            else:
                value_used = default

            sol1 = created_solution_code[sol_id]
            created_solution_code[sol_id] = (
                sol1[: f.start()] + str(value_used) + sol1[f.end() :]
            )

    return [x for x in created_solution_code if x != None]


full_colorshade_list = (
    {color + str(shade) for color in XCOLOR_COLOR_LIST for shade in XCOLOR_SHADE_LIST}
    .union(XCOLOR_COLOR_LIST)
    .union(BW_COLORS)
)


def guess_invalid_choice(choices: list[str]):
    # Detecting numeric lists
    if all([choice.isnumeric() for choice in choices]):
        # detecting color shades of xcolor-material
        int_choices = [int(choice) for choice in choices]
        if all([choice in XCOLOR_SHADE_LIST for choice in int_choices]):
            non_selected_shades = XCOLOR_SHADE_LIST.difference(set(int_choices))
            if len(non_selected_shades) > 0:
                return int(random.choice(list(non_selected_shades)))
            else:
                return None

    # detecting color and shade combos
    if all([choice in full_colorshade_list for choice in choices]):
        non_selected_colors = full_colorshade_list.difference(set(choices))
        if len(non_selected_colors) > 0:
            return random.choice(list(non_selected_colors))
        else:
            return None
    return None


def default_range(template_code):
    return re.sub(r"§range\([^,]+,[^,]+,([^)]+)\)", r"\1", template_code)


def default_rangei(template_code):
    return re.sub(r"§rangei\(([^,]+),[^)]+\)", r"\1", template_code)


def default_choices(template_code):
    return re.sub(r"§choice\(\[[^]]+\],([^)]+)\)", r"\1", template_code)


def get_default(template_code):
    return default_choices(default_range(default_rangei(template_code)))


def generate_all_incorrect_solutions(original_code: str, template_code: str):
    ignored = False
    template_code = handle_def(template_code)[0]
    incorrect_templated_codes = (
        incorrect_from_range(template_code)
        + incorrect_from_rangei(template_code)
        + incorrect_from_choices(template_code)
    )
    if len(incorrect_templated_codes) == 0:
        incorrect_templated_codes = try_generate_incorect_code(original_code, 5)
    if len(incorrect_templated_codes) == 0:
        return [],True
    return [get_default(code) for code in incorrect_templated_codes],ignored
