import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = max(5, 10 * dim)
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
        self.F = 0.8
        self.CR = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, dim))
        self.velocities = np.zeros((self.pop_size, dim))
        self.best_positions = np.copy(self.population)
        self.best_fitness = np.full(self.pop_size, np.inf)
        self.global_best_position = None
        self.global_best_fitness = np.inf

    def __call__(self, func):
        eval_count = 0
        while eval_count < self.budget:
            for i in range(self.pop_size):
                fitness = func(self.population[i])
                eval_count += 1
                if fitness < self.best_fitness[i]:
                    self.best_fitness[i] = fitness
                    self.best_positions[i] = self.population[i]
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = self.population[i]
            
            diversity = np.mean(np.std(self.population, axis=0))
            if diversity < 1e-3:  # Switch to DE if diversity is too low
                for i in range(self.pop_size):
                    indices = list(range(self.pop_size))
                    indices.remove(i)
                    a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                    trial = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                    mask = np.random.rand(self.dim) < self.CR
                    if not np.any(mask):
                        mask[np.random.randint(0, self.dim)] = True
                    self.population[i] = np.where(mask, trial, self.population[i])
            else:  # Use PSO update
                r1, r2 = np.random.rand(2, self.pop_size, self.dim)
                self.velocities = (self.w * self.velocities +
                                   self.c1 * r1 * (self.best_positions - self.population) +
                                   self.c2 * r2 * (self.global_best_position - self.population))
                self.population = np.clip(self.population + self.velocities, self.lower_bound, self.upper_bound)
        
        return self.global_best_position, self.global_best_fitness