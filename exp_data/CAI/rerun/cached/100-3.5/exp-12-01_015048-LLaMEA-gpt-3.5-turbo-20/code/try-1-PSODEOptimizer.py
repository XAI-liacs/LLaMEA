import numpy as np

class PSODEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        def within_bounds(x):
            return np.clip(x, -5.0, 5.0)
        
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        for _ in range(self.budget):
            for i in range(self.budget):
                p_best = population[np.argmin(fitness)]
                g_best = population[np.argmin(fitness)]
                
                v = np.random.uniform() * (p_best - population[i]) + np.random.uniform() * (g_best - population[i])
                candidate = within_bounds(population[i] + v)
                fitness_candidate = func(candidate)
                
                if fitness_candidate < fitness[i]:
                    population[i] = candidate
                    fitness[i] = fitness_candidate
        
        return population[np.argmin(fitness)]