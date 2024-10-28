import random
import numpy as np

class AdaptivePerturbation:
    """
    A heuristic algorithm for black box optimization using adaptive perturbation.
    
    The algorithm uses a combination of random search and perturbation to find the optimal solution.
    It handles a wide range of tasks and can be adapted to various optimization problems.
    """
    
    def __init__(self, budget, dim):
        """
        Initialize the optimizer with a budget and dimensionality.
        
        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = [(-5.0, 5.0)] * dim
        self.func_evaluations = 0
        self.perturbation_threshold = 0.1
        self.perturbation_strategy = "random"

    def __call__(self, func):
        """
        Optimize the black box function using the optimizer.
        
        Args:
            func (function): The black box function to optimize.
        
        Returns:
            tuple: The optimal solution and its cost.
        """
        
        # Initialize the solution and cost
        solution = None
        cost = float('inf')
        
        # Perform random search
        for _ in range(self.budget):
            # Perturb the current solution based on the perturbation strategy
            if self.perturbation_strategy == "random":
                perturbation = (random.uniform(-1, 1), random.uniform(-1, 1))
            elif self.perturbation_strategy == "adaptive":
                if self.func_evaluations < self.budget // 2:
                    perturbation = (random.uniform(-1, 1), random.uniform(-1, 1))
                else:
                    perturbation = (random.uniform(-1, 1), random.uniform(-1, 1)) * self.perturbation_threshold
            else:
                raise ValueError("Invalid perturbation strategy")
            
            # Evaluate the new solution
            new_cost = func(perturbed_solution = perturbation)
            
            # Update the solution and cost if the new solution is better
            if new_cost < cost:
                solution = perturbation
                cost = new_cost
        
        return solution, cost

    def run(self, func, num_iterations):
        """
        Run the optimizer for a specified number of iterations.
        
        Args:
            func (function): The black box function to optimize.
            num_iterations (int): The number of iterations to run.
        
        Returns:
            tuple: The optimal solution and its cost.
        """
        
        # Run the optimizer for the specified number of iterations
        for _ in range(num_iterations):
            solution, cost = self(func)
            self.func_evaluations += 1
            
            # If the optimizer has reached the budget, break the loop
            if self.func_evaluations >= self.budget:
                break
        
        return solution, cost

# Description: Black Box Optimization using Adaptive Perturbation
# Code: 