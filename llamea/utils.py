import ast
import re
from difflib import SequenceMatcher

import numpy as np


class NoCodeException(Exception):
    """Could not extract generated code."""

    pass


def handle_timeout(signum, frame):
    """Raise a timeout exception"""
    raise TimeoutError


def apply_unified_diff(text: str, diff: str) -> str:
    """Apply a unified diff patch to ``text`` and return the modified text.

    This utility parses a unified diff (as produced by ``difflib.unified_diff``)
    and applies the described changes to the given ``text``.

    Args:
        text: The original string to patch.
        diff: The unified diff describing the modifications.

    Returns:
        str: The patched text.
    """

    text_lines = text.splitlines(keepends=True)
    result: list[str] = []
    line_no = 0
    diff_lines = diff.splitlines()
    i = 0
    hunk_re = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

    while i < len(diff_lines):
        line = diff_lines[i]
        match = hunk_re.match(line)
        if match:
            start = int(match.group(1)) - 1
            result.extend(text_lines[line_no:start])
            line_no = start
            i += 1
            while i < len(diff_lines) and not diff_lines[i].startswith("@@"):
                diff_line = diff_lines[i]
                if diff_line.startswith("+"):
                    result.append(diff_line[1:] + "\n")
                elif diff_line.startswith("-"):
                    line_no += 1
                elif diff_line.startswith(" "):
                    result.append(text_lines[line_no])
                    line_no += 1
                i += 1
        else:
            i += 1

    result.extend(text_lines[line_no:])
    return "".join(result)


def discrete_power_law_distribution(n, beta):
    """
    Power law distribution function from:
    # Benjamin Doerr, Huu Phuoc Le, Régis Makhmara, and Ta Duy Nguyen. 2017.
    # Fast genetic algorithms.
    # In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '17).
    # Association for Computing Machinery, New York, NY, USA, 777–784.
    # https://doi.org/10.1145/3071178.3071301
    """

    def discrete_power_law(n, alpha, beta):
        half_n = int(n / 2)
        C_beta_half_n = 0
        for i in range(1, half_n + 1):
            C_beta_half_n += i ** (-beta)
        probability_alpha = C_beta_half_n ** (-1) * alpha ** (-beta)
        return probability_alpha

    half_n = int(n / 2)
    elements = [alpha for alpha in range(1, half_n + 1)]
    probabilities = [discrete_power_law(n, alpha, beta) for alpha in elements]
    if elements == []:
        return 0.05
    else:
        sample = np.random.choice(elements, p=probabilities)
        return sample / n


def code_distance(a, b):
    """Return a rough distance between two solutions based on their ASTs.

    The function accepts either :class:`Solution` objects or raw code strings
    and computes ``1 - similarity`` of their abstract syntax trees using
    :class:`difflib.SequenceMatcher` on the dumped AST representations.
    ``1.0`` is returned on parsing errors or when the inputs cannot be
    processed.

    Args:
        a: The first solution or Python source code.
        b: The second solution or Python source code.

    Returns:
        float: A value in ``[0, 1]`` indicating dissimilarity of the code.
    """

    code_a = getattr(a, "code", a)
    code_b = getattr(b, "code", b)
    try:
        tree_a = ast.parse(code_a)
        tree_b = ast.parse(code_b)
        return 1 - SequenceMatcher(None, ast.dump(tree_a), ast.dump(tree_b)).ratio()
    except Exception:
        return 1.0
