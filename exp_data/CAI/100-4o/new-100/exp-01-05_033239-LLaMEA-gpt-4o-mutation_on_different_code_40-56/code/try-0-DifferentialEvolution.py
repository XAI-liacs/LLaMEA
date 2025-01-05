import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # Adjust population size relative to dimension
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(p) for p in population])
        self.budget -= self.population_size
        
        F, CR = 0.5, 0.9  # Mutation factor and crossover rate
        while self.budget > 0:
            for i in range(self.population_size):
                # Mutation: select three distinct individuals
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = np.clip(x1 + F * (x2 - x3), lb, ub)
                
                # Crossover
                crossover_mask = np.random.rand(self.dim) < CR
                trial = np.where(crossover_mask, mutant, population[i])
                
                # Selection
                trial_fitness = func(trial)
                self.budget -= 1
                
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                
                # Update best solution found
                if fitness[i] < self.f_opt:
                    self.f_opt = fitness[i]
                    self.x_opt = population[i]
                
                if self.budget <= 0:
                    break
        
        return self.f_opt, self.x_opt