import numpy as np
from llamea.solution import Solution
from llamea.multi_objective_fitness import Fitness
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class ParetoArchive:
    def __init__(self, minimisation: bool):
        """
        Generates a Pareto Archive Class, that iteratively takes in a multi-objective solution, and updates the overal
        pareto-front explored by an optimiser.

        ## Params:
        `minimisation: bool`: Assuming all the fitness are either to be minimised or maximised, set this flag to determine the
        direction of optimality.
        """
        self.minimisation = minimisation
        self.archive: list[Solution] = []

    def _clear_to_consider(self, solution: Solution) -> bool:
        """
        Checks if the solution is valid to be considered for pareto-front archive update. If any of the fitness values
        is NaN, then the solution is not considered for pareto-front update.

        ## Args:
        `solution: Solution`: A multi-objective solution that is being considered for pareto-front update.

        ## Returns:
        `consider: bool`: Returns True, if solution is valid to be considered for pareto-front update, else returns False.
        """
        if not isinstance(solution.fitness, Fitness):
            return False

        for key in solution.fitness.keys():
            if np.isnan(solution.fitness[key]):
                return False
            if np.isinf(solution.fitness[key]):
                return False
        return True

    def add_solutions(self, solutions: list[Solution]):
        """
        Updates the pareto-front archive with current solution. First checks for solutions in the
        front that may be dominated. If dominated solutions in front are found, then append the `solution` into
        archive and removed the dominated solutions from archive, else do nothing.

        ## Args:
        `solutions: list[Solution]`: An array of multi-objective solution, that is being added to pareto front archive.
        """

        candidates = [
            individual
            for individual in self.archive + solutions
            if self._clear_to_consider(individual)
        ]

        candidates = list({obj.id: obj for obj in candidates}.values())

        if len(candidates) == 0:
            return

        F = np.array([c.fitness.to_vector() for c in candidates])

        # Handle maximisation by negating objectives
        if not self.minimisation:
            F = -F

        nds = NonDominatedSorting()
        front_idx = nds.do(F, only_non_dominated_front=True)

        self.archive = [candidates[i] for i in front_idx]

    def get_best(self) -> list[Solution]:
        """
        Returns the best multi-objective solutions generated so far; i.e. returns the pareto front of best solution till date.

        ## Args:
        None: No arguements required.

        ## Returns:
        `front: list[Solution]`: Returns a list of solutions that belongs to the pareto-front.
        """
        return self.archive
