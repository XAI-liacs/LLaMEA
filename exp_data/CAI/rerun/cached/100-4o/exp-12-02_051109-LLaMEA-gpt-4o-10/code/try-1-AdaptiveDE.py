import numpy as np

class AdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.lb = -5.0
        self.ub = 5.0
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        
    def __call__(self, func):
        # Initialize a population within bounds
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        
        eval_count = len(population)
        
        while eval_count < self.budget:
            for i in range(self.population_size):
                # Mutation: Select three distinct individuals
                indices = np.random.choice([idx for idx in range(self.population_size) if idx != i], 3, replace=False)
                x1, x2, x3 = population[indices]
                
                # Generate mutant vector
                mutant = x1 + self.F * (x2 - x3)
                
                # Ensure mutant is within bounds
                mutant = np.clip(mutant, self.lb, self.ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                
                # Selection
                trial_fitness = func(trial)
                eval_count += 1
                
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial
                
                if eval_count >= self.budget:
                    break

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]