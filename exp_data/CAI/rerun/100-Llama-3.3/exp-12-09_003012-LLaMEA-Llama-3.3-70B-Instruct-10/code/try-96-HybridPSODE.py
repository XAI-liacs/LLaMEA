import numpy as np

class HybridPSODE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 20
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.8
        self.cr = 0.5
        self.f = 0.5
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))
        self.personal_best = self.population.copy()
        self.personal_best_fitness = np.ones(self.population_size) * np.Inf

    def __call__(self, func):
        for i in range(self.budget):
            for j in range(self.population_size):
                # Calculate fitness
                f = func(self.population[j])
                
                # Update personal best
                if f < self.personal_best_fitness[j]:
                    self.personal_best_fitness[j] = f
                    self.personal_best[j] = self.population[j]
                    
                # Update global best
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = self.population[j]
                    
                # Update velocity and position using PSO with adaptive weight
                self.w = 0.8 - (0.2 * i / self.budget)  # adaptive weight
                self.velocities[j] = self.w * self.velocities[j] + self.c1 * np.random.uniform(0, 1, self.dim) * (self.personal_best[j] - self.population[j]) + self.c2 * np.random.uniform(0, 1, self.dim) * (self.x_opt - self.population[j])
                self.population[j] = self.population[j] + self.velocities[j]
                
                # Apply boundary conditions with reflection
                self.population[j] = np.clip(self.population[j], -5.0, 5.0)
                if np.any(self.population[j] == -5.0) or np.any(self.population[j] == 5.0):
                    self.velocities[j] = -self.velocities[j]  # reflection
                
                # Perform differential evolution
                if np.random.uniform(0, 1) < self.cr:
                    idx = np.random.permutation(self.population_size)
                    idx = idx[idx!= j]
                    r1, r2, r3 = idx[:3]
                    self.population[j] = self.population[r1] + self.f * (self.population[r2] - self.population[r3])
                    self.population[j] = np.clip(self.population[j], -5.0, 5.0)
                    
            # Check if budget is exceeded
            if i >= self.budget - self.population_size:
                break
                
        return self.f_opt, self.x_opt