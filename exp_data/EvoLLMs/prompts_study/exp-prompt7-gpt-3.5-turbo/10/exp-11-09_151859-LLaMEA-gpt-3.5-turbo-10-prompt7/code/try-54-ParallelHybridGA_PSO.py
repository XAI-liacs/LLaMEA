import numpy as np
from joblib import Parallel, delayed

class ParallelHybridGA_PSO:
    def __init__(self, budget, dim, population_size=50, mutation_rate=0.1, inertia_weight=0.5, cognitive_weight=1.5, social_weight=1.5):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        global_best = population[best_idx]
        velocity = np.zeros((self.population_size, self.dim))  # Initialize velocity
        
        for _ in range(0, self.budget, self.population_size):
            batch_fitness = Parallel(n_jobs=-1)(delayed(func)(ind) for ind in population)
            for i, fit in enumerate(batch_fitness):
                if fit < fitness[i]:
                    fitness[i] = fit
                    best_individual = population[i]
            best_idx = np.argmin(fitness)
            global_best = population[best_idx]
            
            velocity = self.inertia_weight * velocity + self.cognitive_weight * np.random.rand() * (best_individual - population) + self.social_weight * np.random.rand() * (global_best - population)
            population += velocity
            
            mutation_mask = np.random.rand(self.population_size, self.dim) < self.mutation_rate
            population = population + np.random.uniform(-1.0, 1.0, (self.population_size, self.dim)) * mutation_mask
        
        return global_best