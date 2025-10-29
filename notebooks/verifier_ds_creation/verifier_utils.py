import math
import random
import re
import numpy as np

from notebooks.verifier_ds_creation.invalid_variant_creation import (
    BW_COLORS,
    XCOLOR_COLOR_LIST,
    XCOLOR_SHADE_LIST,
    apply_random_modifications,
)

random.seed(100)
PER_RANGE_SOLUTIONS = 2
MAX_REG_AMOUNT = 5
isint = lambda v: math.floor(v) == v


def handle_def(template_code: str) -> list[str]:
    return [re.sub(r"§def\((.*?)\)", r"\1", template_code)]


INCORRECT_DISTANCE_RATIO = (
    4 / 10
)  # 4 tenth of the range of possible solution away from the lower and higher limits


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
        if all([isint(float(choice)) for choice in choices]):
            lower = min(int_choices)
            higher = max(int_choices)
            current_distance = INCORRECT_DISTANCE_RATIO * (higher - lower)
            value_used = random.choice(
                [
                    min((lower - current_distance), lower - 1),
                    max((higher + current_distance), higher + 1),
                ]
            )
            return value_used

    # detecting color and shade combos
    if all([choice in full_colorshade_list for choice in choices]):
        non_selected_colors = full_colorshade_list.difference(set(choices))
        if len(non_selected_colors) > 0:
            return random.choice(list(non_selected_colors))
        else:
            return None
    return None


def get_incorrect_for_reg(found_reg: re.Match, code: str):
    streg = found_reg.group()

    if "choice" in streg:
        choices: list = [str(v) for v in eval(found_reg.group(6))]

        value_used = guess_invalid_choice(choices)
        if value_used == None:  # Could not guess an invalid choice
            return None
        return code[: found_reg.start()] + str(value_used) + code[found_reg.end() :]
    else:
        if "rangei" in streg:
            default = float(found_reg.group(4))
            higher = default + float(found_reg.group(5))
            lower = default - float(found_reg.group(5))
            ranges_are_ints = all(
                isint(value)
                for value in [float(found_reg.group(4)), float(found_reg.group(5))]
            )
        elif "range" in streg:
            lower = float(found_reg.group(1))
            higher = float(found_reg.group(2))
            default = float(found_reg.group(3))
            ranges_are_ints = all(isint(value) for value in [lower, higher])
        current_distance = INCORRECT_DISTANCE_RATIO * (higher - lower)
        value_used = random.choice(
            [(lower - current_distance), (higher + current_distance)]
        )  # TODO is it okay to do random?
        return (
            code[: found_reg.start()]
            + (
                str(round(value_used, 2))
                if not ranges_are_ints
                else str(int(value_used))
            )
            + code[found_reg.end() :]
        )


def get_default_for_reg(found_reg: re.Match, code: str):
    streg = found_reg.group()

    if "choice" in streg:
        default = found_reg.group(7)
    else:
        if "rangei" in streg:
            default = found_reg.group(4)
        elif "range" in streg:
            default = found_reg.group(3)
    return code[: found_reg.start()] + (str(default)) + code[found_reg.end() :]


def all_incorrect_from_template(template_code: str) -> list[str]:
    """Generates len(number of ranges)*2 invalid solutions from ranges in template code

    Args:
        template_code (str): The template code

    """

    found_ranges = list(
        re.finditer(
            r"§(?:range\(([^,]+),([^,]+),([^)]+)\)|rangei\(([^,]+),([^)]+)\)|choice\((\[[^]]+\]),([^)]+)\))",
            template_code,
        )
    )
    if len(found_ranges) == 0:
        return []
    if len(found_ranges) > MAX_REG_AMOUNT:
        sampled = sorted(
            random.sample(range(len(found_ranges)), len(found_ranges) - MAX_REG_AMOUNT)
        )
        made_defaults_ranges = [found_ranges[i] for i in sampled]
        for reg in reversed(made_defaults_ranges):
            template_code = get_default_for_reg(reg, template_code)
    found_ranges = list(
        re.finditer(
            r"§(?:range\(([^,]+),([^,]+),([^)]+)\)|rangei\(([^,]+),([^)]+)\)|choice\((\[[^]]+\]),([^)]+)\))",
            template_code,
        )
    )  # recomputed_can be optimized, but not necessary
    arrangements = create_arrangements(
        len(found_ranges), result_amount=-1, true_values_nb=4
    )
    all_resulting_codes = []
    for incorrect_array in arrangements:
        current_code = template_code
        for is_incorrect, found_reg in zip(incorrect_array, reversed(found_ranges)):
            if is_incorrect:
                current_code = get_incorrect_for_reg(found_reg, current_code)
            else:
                current_code = get_default_for_reg(found_reg, current_code)
        all_resulting_codes.append(current_code)
    # trying to paply random modifications
    all_resulting_codes_modified = []
    for code in all_resulting_codes:

        new_modified_code = apply_random_modifications(code, 1, 1, 3)
        if len(new_modified_code)>0:
            all_resulting_codes_modified.append(new_modified_code[0])
        else:
            all_resulting_codes_modified.append(code)
    return all_resulting_codes_modified


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
    incorrect_templated_codes = all_incorrect_from_template(template_code)
    if len(incorrect_templated_codes) == 0:
        incorrect_templated_codes = apply_random_modifications(original_code, 5)
    if len(incorrect_templated_codes) == 0:
        return [], True
    return [get_default(code) for code in incorrect_templated_codes], ignored


from itertools import combinations


def create_arrangements(arr_size, result_amount: int = -1, true_values_nb: int = 2):

    arrangements = []
    true_values_nb = min(arr_size, true_values_nb)

    for combination in combinations(range(arr_size), true_values_nb):
        arr = [0] * arr_size
        for i in combination:
            arr[i] = 1
        arrangements.append(arr)

    if result_amount < 0:
        return arrangements
    else:
        return random.sample(arrangements, result_amount)
