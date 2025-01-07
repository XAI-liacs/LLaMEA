import numpy as np

class HybridQuantumFirefly:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.fireflies = np.random.uniform(size=(self.population_size, dim))
        self.intensities = np.full(self.population_size, np.inf)
        self.global_best = None
        self.global_best_intensity = np.inf
        self.fitness_evaluations = 0
        self.alpha = 0.5
        self.beta_min = 0.2
        self.gamma = 1.0

    def chaotic_map(self, x):
        return 4 * x * (1 - x)

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        lower_bound, upper_bound = bounds[0], bounds[1]
        
        while self.fitness_evaluations < self.budget:
            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break
                
                intensity = func(self.fireflies[i])
                self.fitness_evaluations += 1
                
                if intensity < self.intensities[i]:
                    self.intensities[i] = intensity
                    
                if intensity < self.global_best_intensity:
                    self.global_best_intensity = intensity
                    self.global_best = self.fireflies[i].copy()
            
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if self.intensities[i] > self.intensities[j]:
                        r = np.linalg.norm(self.fireflies[i] - self.fireflies[j])
                        beta = self.beta_min + (1 - self.beta_min) * np.exp(-self.gamma * r ** 2)
                        self.fireflies[i] += beta * (self.fireflies[j] - self.fireflies[i]) + \
                                             self.alpha * (np.random.rand(self.dim) - 0.5)
                        self.fireflies[i] = np.clip(self.fireflies[i], lower_bound, upper_bound)

            self.alpha = self.chaotic_map(self.alpha)
            adaptive_mutation_prob = 0.1 + 0.4 * np.sin(2 * np.pi * self.fitness_evaluations / self.budget)
            
            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break
                
                if np.random.rand() < adaptive_mutation_prob:
                    mutation_vector = lower_bound + np.random.rand(self.dim) * (upper_bound - lower_bound)
                    self.fireflies[i] = mutation_vector

        return self.global_best