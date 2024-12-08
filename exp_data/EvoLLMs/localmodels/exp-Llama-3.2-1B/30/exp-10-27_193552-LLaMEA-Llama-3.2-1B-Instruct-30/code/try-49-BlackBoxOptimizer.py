import random

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
    
    def mutate(self, solution):
        """
        Mutate the current solution.
        
        Args:
            solution (tuple): The current solution.
        
        Returns:
            tuple: The mutated solution.
        """
        
        # Generate a random mutation in the search space
        mutation = (random.uniform(-1, 1), random.uniform(-1, 1))
        
        # Update the solution with the mutation
        solution = (solution[0] + mutation[0], solution[1] + mutation[1])
        
        return solution
    
    def evaluate_fitness(self, solution, func):
        """
        Evaluate the fitness of the current solution using the given function.
        
        Args:
            solution (tuple): The current solution.
            func (function): The black box function to evaluate the fitness.
        
        Returns:
            float: The fitness of the current solution.
        """
        
        # Evaluate the function at the current solution
        fitness = func(solution)
        
        # Update the solution and its fitness
        solution = self.evaluate_individual(solution)
        self.func_evaluations += 1
        
        return fitness
    
    def evaluate_individual(self, solution):
        """
        Evaluate the fitness of the current individual.
        
        Args:
            solution (tuple): The current individual.
        
        Returns:
            float: The fitness of the current individual.
        """
        
        # Evaluate the function at the current individual
        fitness = 0
        for i in range(self.dim):
            fitness += self.search_space[i][0] * solution[i]
        
        # Return the fitness of the current individual
        return fitness
    
    def __str__(self):
        """
        Return a string representation of the optimizer.
        
        Returns:
            str: A string representation of the optimizer.
        """
        
        # Return a string representation of the optimizer
        return f"BlackBoxOptimizer(budget={self.budget}, dim={self.dim})"