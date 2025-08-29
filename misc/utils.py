import numpy as np

try:
    from ioh import LogInfo
    from ioh.logger import AbstractLogger
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
