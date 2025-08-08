import ast
import re
from difflib import SequenceMatcher
from typing import List
import numpy as np
import subprocess

class NoCodeException(Exception):
    """Could not extract generated code."""

    pass


def handle_timeout(signum, frame):
    """Raise a timeout exception"""
    raise TimeoutError


def apply_unified_patch(text: str, diff: str):
    :
    """
    Apply a unified diff to the given text using the system `patch` command.

    This delegates all parsing and application logic to the external `patch`
    utility, which is far more robust than a hand-rolled parser. It handles
    context mismatches, fuzz factors, and edge cases like missing EOF newlines.

    ```text
    ┌─────────────┐
    │   INPUT     │
    │  text:str   │──┐
    └─────────────┘  │
                     ▼
               ┌───────────┐
               │ tempfile  │  → holds original text
               └───────────┘
                     │
                     ▼
              ┌──────────────┐
              │  patch cmd   │ ← receives unified diff on stdin
              └──────────────┘
                     │
                     ▼
               ┌───────────┐
               │ tempfile  │ → now contains patched text
               └───────────┘
                     │
                     ▼
                patched:str
    ```

    Args:
        text: The original text to patch.
        diff: The unified diff (as produced by `git diff`, `difflib.unified_diff`, etc.).
        strip: Optional `-p` value to pass to `patch` (number of path segments to strip).
               Useful if the diff contains file paths you want ignored.

    Returns:
        The patched text as a string.

    Raises:
        subprocess.CalledProcessError: If `patch` fails and returns a nonzero exit code.
        FileNotFoundError: If `patch` is not installed.
    """
    import tempfile
    with tempfile.NamedTemporaryFile("w+", delete=False) as tf:
        tf.write(text)
        tf.flush()
        proc = subprocess.run(
            ["patch", tf.name],
            input=diff.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        tf.seek(0)
        return tf.read()


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
