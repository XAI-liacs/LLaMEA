import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any

class BBOBMetaheuristic:
    """
    A metaheuristic algorithm for solving black box optimization problems.
    """

    def __init__(self, budget: int, dim: int):
        """
        Initialize the algorithm with a given budget and dimensionality.

        Args:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func: Any, bounds: Dict[str, float] = None) -> Any:
        """
        Optimize the given black box function using the provided bounds.

        Args:
        func (Any): The black box function to optimize.
        bounds (Dict[str, float], optional): The bounds for the search space. Defaults to None.

        Returns:
        Any: The optimized function value.
        """
        # Create a copy of the function to avoid modifying the original function
        func_copy = func.copy()

        # Initialize the minimum and maximum function values
        min_func = np.inf
        max_func = -np.inf

        # Initialize the population
        population = np.random.uniform(-bounds["lower_bound"], bounds["upper_bound"], (self.budget, self.dim))

        # Iterate until the population is converged
        while np.all(np.diff(population, axis=0) > 1e-6):
            # Evaluate the population using the provided bounds
            func_values = np.array([func_copy(individual) for individual in population])

            # Select the fittest individuals
            fittest_individuals = np.argmax(func_values, axis=1)

            # Create a new generation by selecting parents and crossover
            new_population = np.array([population[fittest_individuals == i] for i in range(self.budget)])

            # Evaluate the new generation
            func_values_new = np.array([func_copy(individual) for individual in new_population])

            # Update the minimum and maximum function values
            min_func = min(min_func, np.min(func_values_new))
            max_func = max(max_func, np.max(func_values_new))

            # Update the population
            population = new_population

        # Return the optimized function value
        return min_func

# One-line description: A novel metaheuristic algorithm that uses a random search with bounds to optimize black box functions.
# Code: 
# ```python
# import numpy as np
# import scipy.optimize as optimize
#
# def bbobmetaheuristic(budget: int, dim: int) -> float:
#     return optimize.minimize(lambda x: x[0]**2 + x[1]**2, [1, 1], method="SLSQP", bounds=[(-5, 5), (-5, 5)]), budget=budget, dim=dim)