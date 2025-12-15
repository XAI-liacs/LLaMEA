from llamea.solution import Solution

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
        self.archive :list[Solution] = []

    def _dominates(self, a: Solution, b: Solution) -> bool:
        """Returns True if solution `a` dominates solution `b`."""
        v1 = a.get_fitness_vector()
        v2 = b.get_fitness_vector()

        if self.minimisation:
            better_or_equal = all(x <= y for x, y in zip(v1, v2))
            strictly_better = any(x < y for x, y in zip(v1, v2))
        else:
            better_or_equal = all(x >= y for x, y in zip(v1, v2))
            strictly_better = any(x > y for x, y in zip(v1, v2))

        return better_or_equal and strictly_better


    def add_solution(self, solution: Solution):
        """
        Updates the pareto-front archive with current solution. First checks for solutions in the 
        front that may be dominated. If dominated solutions in front are found, then append the `solution` into
        archive and removed the dominated solutions from archive, else do nothing.
        
        ## Args:
        `solution: Solution`: A multi-objective solution, that is being added to pareto front archive.
        """
        assert isinstance(solution.fitness, dict)
        if solution.id in [indv.id for indv in self.archive]:
            return
        
        for m in self.archive:
            if self._dominates(m, solution):
                return  # discard incoming solution

        to_remove = []
        for m in self.archive:
            if self._dominates(solution, m):
                to_remove.append(m)

        for m in to_remove:
            self.archive.remove(m)

        self.archive.append(solution)


    def get_best(self) -> list[Solution]:
        """
        Returns the best multi-objective solutions generated so far; i.e. returns the pareto front of best solution till date.

        ## Args:
        None: No arguements required.

        ## Returns:
        `front: list[Solution]`: Returns a list of solutions that belongs to the pareto-front.
        """
        return self.archive