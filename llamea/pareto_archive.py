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
        self.archive :list[Solution] = []


    def add_solutions(self, solutions: list[Solution]):
        """
        Updates the pareto-front archive with current solution. First checks for solutions in the 
        front that may be dominated. If dominated solutions in front are found, then append the `solution` into
        archive and removed the dominated solutions from archive, else do nothing.
        
        ## Args:
        `solutions: list[Solution]`: An array of multi-objective solution, that is being added to pareto front archive.
        """

        assert all(isinstance(solution.fitness, Fitness) for solution in solutions)
        candidates = list(set(self.archive[:] + solutions))
        
        nds = NonDominatedSorting()

        dominant_index = nds.do(np.array([candidate.fitness.to_vector() for candidate in candidates]), 
               only_non_dominated_front=True)
        
        new_archieve = [candidates[index] for index in dominant_index]

        avaiable_id = [solution.id for solution in self.archive]
        ## Clean recurring solutions.
        new_archieve = list(filter(lambda solution: solution.id not in avaiable_id, new_archieve))
        
        self.archive += new_archieve

    def get_best(self) -> list[Solution]:
        """
        Returns the best multi-objective solutions generated so far; i.e. returns the pareto front of best solution till date.

        ## Args:
        None: No arguements required.

        ## Returns:
        `front: list[Solution]`: Returns a list of solutions that belongs to the pareto-front.
        """
        return self.archive