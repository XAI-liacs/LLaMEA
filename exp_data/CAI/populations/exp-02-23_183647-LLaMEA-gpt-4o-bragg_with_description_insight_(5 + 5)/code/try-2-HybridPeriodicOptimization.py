import numpy as np
from scipy.optimize import minimize

class HybridPeriodicOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def _initialize_population(self, bounds, population_size):
        lb, ub = bounds
        population = np.random.rand(population_size, self.dim) * (ub - lb) + lb
        return population
    
    def _evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])
    
    def _differential_evolution(self, func, bounds, population_size=20, F=0.8, CR=0.9):
        population = self._initialize_population(bounds, population_size)
        fitness = self._evaluate_population(population, func)
        
        for _ in range(self.budget // population_size):
            for i in range(population_size):
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), bounds[0], bounds[1])
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, population[i])
                trial_fitness = func(trial)
                
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
            
            # Enforce periodicity by averaging every two adjacent layers
            half_dim = self.dim // 2
            for ind in population:
                ind[:half_dim] = (ind[:half_dim] + ind[half_dim:]) / 2
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]
    
    def _local_optimization(self, x0, func, bounds):
        result = minimize(func, x0, method='L-BFGS-B', bounds=bounds)
        return result.x, result.fun
    
    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        global_best, global_best_fitness = self._differential_evolution(func, bounds)
        
        # Local optimization to fine-tune the final solution
        local_best, local_best_fitness = self._local_optimization(global_best, func, bounds)
        
        return local_best if local_best_fitness < global_best_fitness else global_best