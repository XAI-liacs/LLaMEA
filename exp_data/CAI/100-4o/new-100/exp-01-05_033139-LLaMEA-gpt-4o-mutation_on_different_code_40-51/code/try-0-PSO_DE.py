import numpy as np

class PSO_DE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.pop_size = 50
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.w = 0.5   # inertia weight
        self.F = 0.8   # differential weight
        self.CR = 0.9  # crossover probability
        self.v_max = (5 - (-5)) / 2  # max velocity
        self.population = np.random.uniform(-5, 5, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-self.v_max, self.v_max, (self.pop_size, self.dim))
        self.p_best = self.population.copy()
        self.p_best_values = np.full(self.pop_size, np.Inf)
        
    def __call__(self, func):
        for i in range(self.budget // self.pop_size):  # Iterating based on population size
            for j in range(self.pop_size):
                f_value = func(self.population[j])
                if f_value < self.p_best_values[j]:
                    self.p_best_values[j] = f_value
                    self.p_best[j] = self.population[j].copy()
                if f_value < self.f_opt:
                    self.f_opt = f_value
                    self.x_opt = self.population[j].copy()
            
            g_best = self.p_best[np.argmin(self.p_best_values)]
            
            for j in range(self.pop_size):
                # PSO update
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.velocities[j] = self.w * self.velocities[j] + self.c1 * r1 * (self.p_best[j] - self.population[j]) + self.c2 * r2 * (g_best - self.population[j])
                self.velocities[j] = np.clip(self.velocities[j], -self.v_max, self.v_max)
                self.population[j] += self.velocities[j]
                
                # DE mutation and crossover
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = self.population[idxs]
                mutant = np.clip(x1 + self.F * (x2 - x3), -5, 5)
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, self.population[j])
                
                # Select between trial vector and current vector
                if func(trial) < func(self.population[j]):
                    self.population[j] = trial

        return self.f_opt, self.x_opt