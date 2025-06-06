

from vif.models.code import CodeEdit


def get_annotated_code(code:str)->str:
    """Returns the code annotated with line numbers"""
    return "\n".join([str(i + 1) + "|" + line for (i, line) in enumerate(code.split("\n"))])


def apply_edits(code:str, edits: list[CodeEdit]):
    edits = sorted(edits, key=lambda a: a.start, reverse=True)

    if any([edits[i].start < edits[i + 1].end for i in range(len(edits) - 1)]):
            raise ValueError("Ranges are overlaping, cancelling edits.")

        # Modifying code
    splitted_code = code.split("\n")
    for edit in edits:
            # -1 because the codeedit specify lines which start at 1, but not +1 for end even splice is [a,b[ because last index replaced

            splitted_code[edit.start - 1 : edit.end - 1] = []
            if edit.content is not None and not edit.content == "":
                splitted_code.insert(edit.start - 1, edit.content)

    code = "\n".join(splitted_code)
    return code