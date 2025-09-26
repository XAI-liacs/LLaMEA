import ast
import re
from difflib import SequenceMatcher
import numpy as np
from difflib import SequenceMatcher


class NoCodeException(Exception):
    """Could not extract generated code."""

    pass


def handle_timeout(signum, frame):
    """Raise a timeout exception"""
    raise TimeoutError


def _code_updater(code: str, lines_to_change: list[str], updated_lines: list[str]):
    """Line by line update code, and return the update.
    Args:
        code: Current code in the individual.
        lines_to_change: A list of lines to be changed by the LLM.
        updated_lines: Lines to replace the `lines_to_update`.

    """
    if len(lines_to_change) != len(lines_to_change):
        raise ValueError
    for i in range(len(lines_to_change)):
        code = code.replace(
            lines_to_change[i], updated_lines[i], 1
        )  # Update one occurance of lines_to_change, to corresponding change.
    return code


def apply_code_delta(text: str, base_code: str) -> tuple[str, bool, float]:
    """
    Assuming the LLM follows the intructions properly, following format of response is expected.
    ```diff <- (diff may appear sometimes.)
    # A series of following search replace pattern will appear.
    <<<<<<< SEARCH
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    =======
    # Reorder loops for better memory access pattern
    for i in range(m):
        for k in range(n):
            for j in range(p):
                C[i, j] += A[i, k] * B[k, j]
    >>>>>>> REPLACE
    ```

    Args:
        text: LLM response.text.
        base_code: Base code to be mutated.
    Returns:
        Code: updated code, after applying diff.
        bool: Success of diff mode implementation.
        float: Ratio of code changed.
    """
    outLines = []
    inLines = []
    try:
        pattern = re.compile(
            r"(?s)<{3,}\s*SEARCH\s*\n(.*?)\n={3,}\s*\n(.*?)(?=\n>{3,}\s*REPLACE)"
        )
        matches = pattern.findall(text)
        if len(matches) == 0:
            print(
                "WARNING: LLM didn't adhere to search replace pattern. Try bigger model."
            )
            raise ValueError

        for search, replace in matches:
            outLines.append(search)
            inLines.append(replace)

        code = _code_updater(base_code, outLines, inLines)

        seq_match = SequenceMatcher(None, code, base_code)
        ratio = seq_match.ratio()

        return code, True, ratio

    except Exception:
        return base_code, False, 1.0


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
