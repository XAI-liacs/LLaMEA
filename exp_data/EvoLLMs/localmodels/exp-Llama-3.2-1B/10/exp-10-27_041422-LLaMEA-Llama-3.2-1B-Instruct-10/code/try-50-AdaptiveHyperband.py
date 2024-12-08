import numpy as np
import os

class AdaptiveHyperband:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_func = None
        self.best_func_evals = 0
        self.sample_size = 1
        self.sample_dir = None
        self.sample_history = []
        self.new_individual_history = []

    def __call__(self, func, iterations):
        if self.budget <= 0:
            raise ValueError("Budget cannot be zero or negative")
        
        if self.best_func is not None:
            return self.best_func
        
        # Initialize the best function and its evaluation count
        self.best_func = func
        self.best_func_evals = 1
        
        # Set the sample size and directory
        self.sample_size = 10
        self.sample_dir = f"sample_{self.sample_size}"
        
        # Initialize the population with random individuals
        self.population = self.generate_population(self.sample_size, self.dim)
        
        # Perform adaptive sampling
        for _ in range(iterations):
            # Generate a random sample of size self.sample_size
            sample = np.random.uniform(-5.0, 5.0, size=self.dim)
            
            # Evaluate the function at the current sample
            func_eval = func(sample)
            
            # If this is the first evaluation, update the best function
            if self.best_func_evals == 1:
                self.best_func = func_eval
                self.best_func_evals = 1
            # Otherwise, update the best function if the current evaluation is better
            else:
                if func_eval > self.best_func:
                    self.best_func = func_eval
                    self.best_func_evals = 1
                else:
                    self.best_func_evals += 1
            
            # Save the current sample to the sample directory
            self.sample_history.append(sample)
        
        return self.best_func

    def generate_population(self, size, dim):
        # Generate random individuals with a specified dimension
        return np.random.uniform(-5.0, 5.0, size=(size, dim)).tolist()

    def evaluate_fitness(self, individual, iterations):
        # Evaluate the fitness of an individual using the BBOB function
        func_eval = self.budget * individual
        return func_eval

    def save_population(self, population):
        # Save the population to a file
        np.save(f"{self.sample_dir}_population.npy", population)