import re
import os
import ast
import importlib
import jsonlines
import numpy as np
from typing import Any, Optional
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
        float: Ratio of new code similar to the original `base_code`.
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


def _collect_imports(code: str):
    """Collect import info from code using AST.

    Args:
        `code: str` The source code as a string.
    Returns:
        `imports: [{str: str | None}]`: A list of import symbols, containing import type "from" | "import",
            module name in "module", sub module name in "name", and alias name-followed by `as` keyword--in "alias".
    """
    tree = ast.parse(code)
    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    {"type": "import", "module": alias.name, "alias": alias.asname}
                )
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.append(
                    {
                        "type": "from",
                        "module": node.module,
                        "name": alias.name,
                        "alias": alias.asname,
                    }
                )
    return imports


def _add_builtins_into(allowed_list: list[str]):
    """
    Adds safe `__builtins__` library to allowed_list.

    Args:
        `allowed_list: list[str]: ` A list of allowed libraries, that are pip installable.

    Returns:
        `None` (Uses reference semantics to add `__builtins__` to `allowed_list`).
    """
    allowed_list += ["math", "random", "statistics", "itertools", "operator", "heapq"]


def prepare_namespace(
    code: str, allowed: list[str], logger: Any = None
) -> tuple[dict[str, Any], Optional[str]]:
    """Prepare exec global_namespace, with the libraries imported in the text, `code` parameter accepts.
        If the imports are not allowed in the environment, a generic object is provided.

    ### Args:
        `code: str`: Code parameter that is to be passed to `exec` function.

        `allowed: list[str]`: A list of allowed pip installable libraries, that are acceptable to be imported.

        `logger: Any`: Logger with `log_import_fail(list[str])` method in it, LLaMEA has this feature in llamea.loggers.ExperimentLogger.

    ### Returns:
        Returns a prepared global_namespace dictionary for exec, of type `dict[str, Any]`, along with an str,
        `potential_issue`, which can be passed out to feedback to LLM when `exec` throws.

    """
    ns = {}
    imports = _collect_imports(code)

    allowed = allowed.copy()
    allowed = list(map(lambda x: x.split(">")[0], allowed))
    _add_builtins_into(allowed)
    not_allowed: list[str] = []

    for imp in imports:
        if imp["type"] == "import":
            module = imp["module"]

            if allowed and not any(
                module == a or module.startswith(a + ".") for a in allowed
            ):
                ns[imp["alias"] or module.split(".")[0]] = object
                not_allowed.append(imp["module"])
            else:
                mod = importlib.import_module(module)
                ns[imp["alias"] or module.split(".")[0]] = mod

        elif imp["type"] == "from":
            module = imp["module"]

            if allowed and not any(
                module == a or module.startswith(a + ".") for a in allowed
            ):
                ns[imp["alias"] or imp["name"]] = object
                not_allowed.append(imp["module"])
            else:
                mod = importlib.import_module(module)
                obj = getattr(mod, imp["name"])
                ns[imp["alias"] or imp["name"]] = obj

    potential_issue = None

    if logger:
        try:
            logger.log_import_fails(not_allowed)
        except Exception as e:
            print("Provided logger doesn't have log_import_fail", e.__repr__())

    if len(not_allowed) > 0:
        potential_issue = (
            ", ".join(not_allowed)
            + f" {'are' if len(not_allowed) > 1 else 'is'} currently not allowed to be imported in this framework."
        )
    return (ns, potential_issue)


def clean_local_namespace(
    local_namespace: dict[str, Any], global_namespace: dict[str, Any]
):
    """The exec command upon execution, adds global_namespace parameters to local_namespace parameters.
    This function returns local_ns - gobal_ns, so that sweeping for object type never returns a library imported objects.

    Args:
        `local_namespace : dict[str, Any]`: Dictionary that was passed as local_namespace to `exec` block.
        `global_namespace : dict[str, Any]`: Dictionary/Mapping passed as global_namespace to `exec` block.

    Returns:
        Original `local_namespace`, that is `local_namespace` - `global_namespace`.
    """
    for key in global_namespace:
        if key in local_namespace:
            local_namespace.pop(key)
    return local_namespace
