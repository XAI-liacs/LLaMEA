import numpy as np
import random

class MetaGradientDescent:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the meta-gradient descent algorithm.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.noise_level = noise_level
        self.noise = 0

    def __call__(self, func):
        """
        Optimize the black box function `func` using meta-gradient descent.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Initialize the parameter values to random values within the search space
        self.param_values = np.random.uniform(-5.0, 5.0, self.dim)

        # Accumulate noise in the objective function evaluations
        for _ in range(self.budget):
            # Evaluate the objective function with accumulated noise
            func_value = func(self.param_values + self.noise * np.random.normal(0, 1, self.dim))

            # Update the parameter values based on the accumulated noise
            self.param_values += self.noise * np.random.normal(0, 1, self.dim)

        # Refine the solution by changing the individual lines of the selected solution
        # to refine its strategy
        self.param_values = self.refine_solution(self.param_values, func)

        # Return the optimized parameter values and the objective function value
        return self.param_values, func(self.param_values)

    def refine_solution(self, individual, func):
        """
        Refine the solution by changing the individual lines of the selected solution
        to refine its strategy.

        Args:
            individual (numpy.ndarray): The current solution.
            func (callable): The black box function.

        Returns:
            numpy.ndarray: The refined solution.
        """
        # Select the individual lines of the selected solution to refine
        # to minimize the objective function value
        idx = np.random.choice(len(individual), size=dim, replace=False)
        idx = np.sort(idx)

        # Select the lines to mutate
        lines_to_mutate = idx[:int(0.2 * len(idx))]

        # Mutate the lines to refine the solution
        mutated_lines = individual[:lines_to_mutate] + individual[lines_to_mutate + 1:]

        # Refine the solution by applying the mutation
        refined_individual = mutated_lines + individual[lines_to_mutate]

        return refined_individual

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using meta-gradient descent
# to refine the solution by changing individual lines of the selected solution
# to minimize the objective function value