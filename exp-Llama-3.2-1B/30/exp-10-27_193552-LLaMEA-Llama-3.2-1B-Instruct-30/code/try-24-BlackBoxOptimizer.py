# Description: Randomized Black Box Optimization Algorithm (RBBOA)
# Code: 
# ```python
import random
import numpy as np

class BlackBoxOptimizer:
    """
    A metaheuristic algorithm to optimize black box functions.
    
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
            # Perturb the current solution
            perturbed_solution = self.perturb(solution)
            
            # Evaluate the new solution
            new_cost = func(perturbed_solution)
            
            # Update the solution and cost if the new solution is better
            if new_cost < cost:
                solution = perturbed_solution
                cost = new_cost
        
        return solution, cost
    
    def perturb(self, solution):
        """
        Perturb the current solution.
        
        Args:
            solution (tuple): The current solution.
        
        Returns:
            tuple: The perturbed solution.
        """
        
        # Generate a random perturbation in the search space
        perturbation = (random.uniform(-1, 1), random.uniform(-1, 1))
        
        # Update the solution with the perturbation
        solution = (solution[0] + perturbation[0], solution[1] + perturbation[1])
        
        return solution
    
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

def randomized_black_box_optimization(budget, dim, func):
    """
    A novel metaheuristic algorithm for black box optimization.
    
    The algorithm uses a combination of random search and perturbation to find the optimal solution.
    It handles a wide range of tasks and can be adapted to various optimization problems.
    
    Parameters:
    budget (int): The maximum number of function evaluations allowed.
    dim (int): The dimensionality of the search space.
    func (function): The black box function to optimize.
    
    Returns:
    tuple: The optimal solution and its cost.
    """
    
    # Initialize the optimizer with a budget and dimensionality
    optimizer = BlackBoxOptimizer(budget, dim)
    
    # Run the optimizer for a specified number of iterations
    solution, cost = optimizer.run(func, 1000)
    
    # Refine the solution using a probability of 0.3 to change the individual lines of the selected solution
    refined_solution = (solution[0] + 0.3 * (solution[0] - 5.0), solution[1] + 0.3 * (solution[1] - 5.0))
    
    # Evaluate the refined solution
    refined_cost = func(refined_solution)
    
    # Update the solution and cost if the refined solution is better
    if refined_cost < cost:
        solution = refined_solution
        cost = refined_cost
    
    return solution, cost

# Example usage:
def test_function(x):
    return x**2 + 2*x + 1

# Run the optimizer
solution, cost = randomized_black_box_optimization(100, 10, test_function)

# Print the results
print(f"Optimal solution: {solution}")
print(f"Optimal cost: {cost}")