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


def _replace_colors(code: str) -> str:
    used_collist = XCOLOR_COLOR_LIST if "{xcolor-material}" in code else XCOLOR_LIST
    replacements = 0  # counter

    def repl(m):
        nonlocal replacements
        if replacements >= 3:
            return m.group(0)

        prefix, cur = m.groups()
        text_after = m.string[m.end() :]
        next_line = text_after.split("\n", 1)[1] if "\n" in text_after else ""
        if r"\pic" in next_line:
            return m.group(0)

        candidates = [c for c in used_collist if c.lower() != cur.lower()]
        if not candidates:
            return m.group(0)

        replacements += 1
        return prefix + random.choice(candidates)

    return _COLOR_RE.sub(repl, code)


positions = ("below right", "below left", "above right", "above left")
_POS_RE = re.compile(r"\b(" + "|".join(re.escape(p) for p in positions) + r")\b")


def _tweak_position(code: str) -> str:
    replacements = 0

    def repl(m):
        nonlocal replacements
        if replacements >= 2:
            return m.group(0)
        cur = m.group(1)
        candidates = [p for p in positions if p != cur]
        replacements += 1
        return random.choice(candidates)

    return _POS_RE.sub(repl, code)


def _tweak_existing_rotates(code: str) -> str:
    return re.sub(
        r"(rotate\s*=\s*)(-?\d+)",
        lambda m: f"{m.group(1)}{int(m.group(2)) + random.randint(-15, 15)}",
        code,
    )


def _add_rotate_or_scale_to_one_scope(code: str) -> str:
    # find \begin{scope}[...]? occurrences
    pattern = re.compile(r"\\begin\{scope\}(\[.*?\])?", re.DOTALL)
    matches = list(pattern.finditer(code))
    if not matches:
        return code
    candidates = [
        m
        for m in matches
        if (m.group(1) is None)
        or ("rotate" not in m.group(1) and "scale" not in m.group(1))
    ]
    if not candidates:
        return code
    target = random.choice(candidates)
    start, end = target.span()
    scope_opts = target.group(1)
    if random.choice([True, False]):
        new_attr = f"rotate={random.randint(-25,25)}"
    else:
        new_attr = f"scale={round(random.uniform(0.6, 1.4), 2)}"
    if scope_opts:
        # merge new attr into existing bracket
        new_scope = re.sub(
            r"\[(.*?)\]",
            lambda m: f"[{m.group(1)}, {new_attr}]",
            target.group(0),
            count=1,
        )
    else:
        new_scope = target.group(0) + f"[{new_attr}]"
    return code[:start] + new_scope + code[end:]


def _modify_few_numbers(code: str, min_k: int = 2, max_k: int = 3) -> str:
    """Select 2-3 safe numeric tokens and modify them (not touching colors like Blue500)."""
    matches = list(_NUM_RE.finditer(code))
    if not matches:
        return code
    k = min(len(matches), random.randint(min_k, max_k))
    chosen = set(random.sample(range(len(matches)), k))

    parts: List[str] = []
    last_i = 0
    for idx, m in enumerate(matches):
        s, e = m.span(1)  # span of the numeric group
        if idx in chosen:
            parts.append(code[last_i:s])
            orig_num = m.group(1)
            parts.append(_perturb_number_str(orig_num))
            last_i = e
    parts.append(code[last_i:])
    return "".join(parts)


renderer = TexRenderer()

import imagehash


def image_sim(im1, im2):

    with torch.no_grad(), torch.autocast("cuda"):
        im1_hash = imagehash.phash(im1)
        im2_hash = imagehash.phash(im2)

    return im1_hash - im2_hash


MAX_ATTEMPTS = 200


def try_generate_incorect_code(tikz_code: str, num_code: int = 1):
    variants: List[str] = []
    original_image = renderer.from_string_to_image(tikz_code)
    i = 0
    attempt_nb = 0
    original_image.save("./.oui/original.png")

    while len(variants) < num_code and attempt_nb < MAX_ATTEMPTS:
        code = tikz_code

        # 1) Replace a few color attributes with a different color from the palette.
        code = _replace_colors(code)
        # 2) Tweak existing rotate=... values lightly
        code = _tweak_existing_rotates(code)
        # 3) Modify 2-3 numeric tokens (safe selection)
        code = _modify_few_numbers(code, min_k=2, max_k=3)
        # 4) Add rotate/scale to one scope that doesn't already have them
        code = _add_rotate_or_scale_to_one_scope(code)
        # 5) tweak positions
        code = _tweak_position(code)

        attempt_nb += 1
        logger.info(attempt_nb)

        try:
            open(f"./.oui/{i}.tex", "w").write(code)
            image = renderer.from_string_to_image(code)
            # filtering images of differnet sizes(makes things simpler)
            image.save(f"./.oui/{i}.png")
            if (
                image_sim(image, original_image) <= 18
            ):  # after manual tests, 18 seems to be a sweet spot
                continue
            variants.append(code)
            i += 1
        except TexRendererException as t:
            continue

    return variants
