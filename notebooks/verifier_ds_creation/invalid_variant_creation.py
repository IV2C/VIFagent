import random
import re
from typing import List

from loguru import logger
import open_clip
import torch

from vif.utils.image_utils import nmse
from vif.utils.renderer.tex_renderer import TexRenderer, TexRendererException

XCOLOR_SHADE_LIST = {50, 100, 200, 300, 400, 500, 600, 700, 800, 900}

XCOLOR_COLOR_LIST = {
    "Red",
    "Pink",
    "Purple",
    "Deep",
    "Purple",
    "Indigo",
    "Blue",
    "LightBlue",
    "Cyan",
    "Teal",
    "Green",
    "LightGreen",
    "Lime",
    "Yellow",
    "Amber",
    "Orange",
    "DeepOrange",
    "Brown",
    "Grey",
    "BlueGrey",
}

XCOLOR_LIST = {
    "red",
    "green",
    "blue",
    "cyan",
    "magenta",
    "yellow",
    "black",
    "gray",
    "white",
    "darkgray",
    "lightgray",
    "brown",
    "lime",
    "olive",
    "orange",
    "pink",
    "purple",
    "teal",
    "violet",
}

BW_COLORS = {"Black", "White"}


_NUM_RE = re.compile(r"(?<![A-Za-z0-9_#])([-+]?\d*\.?\d+)(?![A-Za-z0-9_])")
_COLOR_RE = re.compile(r"(fill=|draw=|color=)([A-Za-z]+)")


def _perturb_number_str(s: str) -> str:
    """Perturb numeric string s more aggressively while keeping format (int/float)."""
    if "." in s or "e" in s.lower():
        val = float(s)
        # larger multiplicative perturbation and some additive noise
        factor = random.uniform(0.8, 1.2)
        add = random.uniform(-0.2, 0.2) * abs(val)
        new = val * factor + add
        # keep 2 decimal places for floats
        return f"{new:.2f}"
    else:
        val = int(s)
        # for integers, either scale or add a moderate offset
        if abs(val) <= 10:
            new = val + random.randint(-5, 5)
        else:
            factor = random.uniform(0.5, 1.5)
            add = random.uniform(-0.1, 0.1) * abs(val)
            new = int(round(val * factor + add))
        return str(new)


def _replace_colors(code: str) -> tuple[str, bool]:
    used_collist = XCOLOR_COLOR_LIST if "{xcolor-material}" in code else XCOLOR_LIST
    modified = False

    def repl(m):
        nonlocal modified
        if modified:  # only one replacement
            return m.group(0)

        prefix, cur = m.groups()
        text_after = m.string[m.end() :]
        next_line = text_after.split("\n", 1)[1] if "\n" in text_after else ""
        if r"\pic" in next_line:
            return m.group(0)

        candidates = [c for c in used_collist if c.lower() != cur.lower()]
        if not candidates:
            return m.group(0)

        modified = True
        return prefix + random.choice(candidates)

    new_code = _COLOR_RE.sub(repl, code)
    return new_code, modified


def _swap_two_random_node_usages(code: str) -> tuple[str, bool]:
    pat = re.compile(r"\\node(?:\[[^\]]*\])?\s*\(\s*([^)]+?)\s*\)")
    defs = [m for m in pat.finditer(code)]
    if len(defs) < 2:
        return code, False

    a, b = random.sample([m.group(1).strip() for m in defs], 2)

    # Find last definition index of either node
    last_idx = max((m.end() for m in defs if m.group(1).strip() in (a, b)), default=0)
    before, after = code[:last_idx], code[last_idx:]

    # Find all occurrences of (a) or (b) in the after part
    occ = [m for m in re.finditer(rf"\b({re.escape(a)}|{re.escape(b)})\b", after)]
    if len(occ) < 2:
        return code, False

    chosen = random.sample(occ, 2)
    chars = list(after)
    mapping = {a: b, b: a}

    # Replace chosen matches (in reverse to not mess up indices)
    for m in sorted(chosen, key=lambda x: x.start(), reverse=True):
        s, e = m.span()
        chars[s:e] = mapping[m.group(1)]

    new_code = before + "".join(chars)
    return new_code, new_code != code


positions = [
    "below right",
    "below left",
    "above right",
    "above left",
    "above",
    "right",
    "below",
    "left",
]
_POS_RE = re.compile(r"\b(" + "|".join(re.escape(p) for p in positions) + r")\b")


def _tweak_position(code: str) -> tuple[str, bool]:
    modified = False

    def repl(m):
        nonlocal modified
        if modified:  # only one replacement allowed
            return m.group(0)

        cur = m.group(1)
        candidates = [p for p in positions if p != cur]
        if not candidates:
            return m.group(0)

        modified = True
        return random.choice(candidates)

    new_code = _POS_RE.sub(repl, code)
    return new_code, modified


def _tweak_existing_rotates(code: str) -> tuple[str, bool]:
    modified = False

    def repl(m):
        nonlocal modified
        if modified:  # only one modification
            return m.group(0)

        modified = True
        new_val = int(m.group(2)) + random.randint(-15, 15)
        return f"{m.group(1)}{new_val}"

    new_code = re.sub(r"(rotate\s*=\s*)(-?\d+)", repl, code)
    return new_code, modified


def _add_rotate_or_scale_to_one_scope(code: str) -> tuple[str, bool]:
    modified = False
    pattern = re.compile(r"\\begin\{scope\}(\[.*?\])?", re.DOTALL)
    matches = list(pattern.finditer(code))
    if not matches:
        return code, modified

    candidates = [
        m
        for m in matches
        if (m.group(1) is None)
        or ("rotate" not in m.group(1) and "scale" not in m.group(1))
    ]
    if not candidates:
        return code, modified

    target = random.choice(candidates)
    start, end = target.span()
    scope_opts = target.group(1)

    if random.choice([True, False]):
        new_attr = f"rotate={random.randint(-25, 25)}"
    else:
        new_attr = f"scale={round(random.uniform(0.6, 1.4), 2)}"

    if scope_opts:
        new_scope = re.sub(
            r"\[(.*?)\]",
            lambda m: f"[{m.group(1)}, {new_attr}]",
            target.group(0),
            count=1,
        )
    else:
        new_scope = target.group(0) + f"[{new_attr}]"

    modified = True
    return code[:start] + new_scope + code[end:], modified


def _modify_one_number(code: str) -> tuple[str, bool]:
    """Select one safe numeric token and modify it (not touching colors like Blue500)."""
    matches = list(_NUM_RE.finditer(code))
    if not matches:
        return code, False

    target = random.choice(matches)
    s, e = target.span(1)  # span of the numeric group

    new_code = code[:s] + _perturb_number_str(target.group(1)) + code[e:]
    return new_code, True


renderer = TexRenderer()

import imagehash


def image_sim(im1, im2):

    with torch.no_grad(), torch.autocast("cuda"):
        im1_hash = imagehash.phash(im1)
        im2_hash = imagehash.phash(im2)

    return im1_hash - im2_hash




def apply_random_modifications(tikz_code: str, num_modifications=5, outputed_codes=1,max_attempts=15):
    open(f"./.oui/original.tex", "w").write(tikz_code)
    original_image = renderer.from_string_to_image(tikz_code)
    # original_image.save("./.oui/original.png")
    modification_functions = [
        _replace_colors,
        _tweak_existing_rotates,
        _modify_one_number,
        _add_rotate_or_scale_to_one_scope,
        _tweak_position,
        _swap_two_random_node_usages,
    ]
    random.shuffle(modification_functions)
    modification_functions = random.sample(
        modification_functions, len(modification_functions) - 1
    )  # removing a random modification function to try to broader the set of modifications

    candidate_codes: list[float, str] = []

    attempt_number = 0
    while attempt_number < max_attempts:
        n_modif = 0
        current_code = tikz_code
        while n_modif < num_modifications:
            for mod_function in modification_functions:
                current_code, applied = mod_function(current_code)
                if applied:
                    n_modif += 1
                if n_modif >= num_modifications:
                    break

        try:
            # open(f"./.oui/{attempt_number}.tex", "w").write(current_code)
            image = renderer.from_string_to_image(current_code)
            # image.save(f"./.oui/{attempt_number}.png")
            candidate_codes.append((image_sim(image, original_image), current_code))
            attempt_number += 1
        except TexRendererException as t:
            attempt_number += 1

    return [
        code for _, code in sorted(candidate_codes, key=lambda k: k[0], reverse=True)
    ][outputed_codes:]
