import numpy as np

class FAwGE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        num_fireworks = 10
        num_sparks = 5
        sparks_factor = 0.1
        
        fireworks_positions = np.random.uniform(low=-5.0, high=5.0, size=(num_fireworks, self.dim))
        fireworks_fitness = np.array([func(firework) for firework in fireworks_positions])
        
        for _ in range(self.budget):
            sparks_positions = np.array([firework + np.random.normal(scale=sparks_factor, size=self.dim) for _ in range(num_sparks) for firework in fireworks_positions])
            sparks_fitness = np.array([func(spark) for spark in sparks_positions])
            
            combined_positions = np.concatenate((fireworks_positions, sparks_positions), axis=0)
            combined_fitness = np.concatenate((fireworks_fitness, sparks_fitness), axis=0)
            sorted_indices = np.argsort(combined_fitness)
            
            fireworks_positions = combined_positions[sorted_indices[:num_fireworks]]
            fireworks_fitness = combined_fitness[sorted_indices[:num_fireworks]]
        
        best_position = fireworks_positions[np.argmin(fireworks_fitness)]
        return best_position