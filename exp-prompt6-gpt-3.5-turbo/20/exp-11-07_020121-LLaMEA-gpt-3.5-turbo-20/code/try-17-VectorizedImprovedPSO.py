import numpy as np

class VectorizedImprovedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.w = 0.9  # Initial inertia weight
        self.c1 = 1.496
        self.c2 = 1.496
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.velocities = np.zeros((budget, dim))

    def __call__(self, func):
        fitness_values = [func(x) for x in self.population]
        g_best_idx = np.argmin(fitness_values)
        g_best = self.population[g_best_idx]
        
        for _ in range(self.budget):
            r1, r2 = np.random.rand(), np.random.rand()
            self.velocities = self.w * self.velocities + self.c1 * r1 * (self.population - self.population) + self.c2 * r2 * (g_best - self.population)
            np.clip(self.velocities, -5.0, 5.0, out=self.velocities)  # Clip velocities directly
            self.population += self.velocities  # Update positions efficiently
            
            new_fitness_values = [func(x) for x in self.population]
            new_g_best_idx = np.argmin(new_fitness_values)
            better_indices = new_fitness_values < fitness_values
            g_best_idx = np.where(better_indices, new_g_best_idx, g_best_idx)
            g_best = self.population[g_best_idx]
            fitness_values = np.where(better_indices, new_fitness_values, fitness_values)
        
        return g_best