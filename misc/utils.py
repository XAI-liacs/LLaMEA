from venv import logger
import numpy as np
import re

from difflib import SequenceMatcher

try:
    from ioh import LogInfo
    try:
        from ioh.logger import AbstractLogger
    except ImportError:
        from ioh import logger as iohLogger
        AbstractLogger = iohLogger.AbstractLogger
except ImportError:
    LogInfo = None
    AbstractLogger = object




class ThresholdReachedException(Exception):
    """The algorithm reached the lower threshold."""

    pass


class OverBudgetException(Exception):
    """The algorithm tried to do more evaluations than allowed."""

    pass


def correct_aoc(ioh_function, logger, budget):
    """Correct aoc values in case a run stopped before the budget was exhausted

    Args:
        ioh_function: The function in its final state (before resetting!)
        logger: The logger in its final state, so we can ensure the settings for aoc calculation match
        budget: The intended maximum budget

    Returns:
        float: The normalized aoc of the run, corrected for stopped runs
    """
    fraction = (
        logger.transform(
            np.clip(
                ioh_function.state.current_best_internal.y, logger.lower, logger.upper
            )
        )
        - logger.transform(logger.lower)
    ) / (logger.transform(logger.upper) - logger.transform(logger.lower))
    aoc = (
        logger.aoc
        + np.clip(budget - ioh_function.state.evaluations, 0, budget) * fraction
    ) / budget

    return 1 - aoc


class aoc_logger(AbstractLogger):
    """aoc_logger class implementing the logging module for ioh."""

    def __init__(
        self,
        budget,
        lower=1e-8,
        upper=1e8,
        scale_log=True,
        stop_on_threshold=False,
        *args,
        **kwargs,
    ):
        """Initialize the logger.

        Args:
            budget (int): Evaluation budget for calculating aoc.
        """
        super().__init__(*args, **kwargs)
        self.aoc = 0
        self.lower = lower
        self.upper = upper
        self.budget = budget
        self.stop_on_threshold = stop_on_threshold
        self.transform = lambda x: np.log10(x) if scale_log else (lambda x: x)

    def __call__(self, log_info: LogInfo):
        """Subscalculate the aoc.

        Args:
            log_info (ioh.LogInfo): info about current values.
        """
        if log_info.evaluations > self.budget:
            raise OverBudgetException
        if log_info.evaluations == self.budget:
            return
        if self.stop_on_threshold and abs(log_info.raw_y_best) < self.lower:
            raise ThresholdReachedException
        y_value = np.clip(log_info.raw_y_best, self.lower, self.upper)
        self.aoc += (self.transform(y_value) - self.transform(self.lower)) / (
            self.transform(self.upper) - self.transform(self.lower)
        )

    def reset(self, func):
        super().reset()
        self.aoc = 0


class budget_logger(AbstractLogger):
    """budget_logger class implementing the logging module for ioh."""

    def __init__(
        self,
        budget,
        *args,
        **kwargs,
    ):
        """Initialize the logger.

        Args:
            budget (int): Evaluation budget for calculating aoc.
        """
        super().__init__(*args, **kwargs)
        self.budget = budget

    def __call__(self, log_info: LogInfo):
        """Subscalculate the aoc.

        Args:
            log_info (ioh.LogInfo): info about current values.
        """
        if log_info.evaluations > self.budget:
            raise OverBudgetException

    def reset(self):
        super().reset()


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
