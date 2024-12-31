import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.5
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 20
        self.bounds = (-5.0, 5.0)

    def __call__(self, func):
        lb, ub = self.bounds
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))
        personal_best_positions = np.copy(self.population)
        personal_best_scores = np.full(self.population_size, np.Inf)
        
        for _ in range(self.budget // self.population_size):
            for i in range(self.population_size):
                f = func(self.population[i])
                if f < personal_best_scores[i]:
                    personal_best_scores[i] = f
                    personal_best_positions[i] = self.population[i]
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = self.population[i]

            # Update particle velocities and positions
            r1, r2 = np.random.rand(2)
            self.velocities = (self.w * self.velocities + 
                               self.c1 * r1 * (personal_best_positions - self.population) + 
                               self.c2 * r2 * (self.x_opt - self.population))
            self.population += self.velocities
            self.population = np.clip(self.population, lb, ub)
            
            # Adaptive parameter tuning
            self.w = max(0.2, self.w * 0.99)
            self.c1, self.c2 = 1.5 + 0.5 * np.random.rand(), 1.5 + 0.5 * np.random.rand()
            
        return self.f_opt, self.x_opt