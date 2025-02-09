import numpy as np

class HybridQuantumPSO:
    def __init__(self, budget, dim, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.positions = np.random.uniform(size=(swarm_size, dim))
        self.velocities = np.random.uniform(size=(swarm_size, dim))
        self.best_positions = self.positions.copy()
        self.best_scores = np.full(swarm_size, np.inf)
        self.global_best_position = np.zeros(dim)
        self.global_best_score = np.inf
        self.w = 0.5  # inertia weight
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        
    def levy_flight(self, lambda_val=1.5):
        sigma = (np.math.gamma(1 + lambda_val) * np.sin(np.pi * lambda_val / 2) /
                 (np.math.gamma((1 + lambda_val) / 2) * lambda_val * 2 ** ((lambda_val - 1) / 2))) ** (1 / lambda_val)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        return u / np.abs(v) ** (1 / lambda_val)
        
    def optimize(self, func):
        num_evaluations = 0
        
        while num_evaluations < self.budget:
            for i in range(self.swarm_size):
                self.positions[i] = np.clip(self.positions[i], func.bounds.lb, func.bounds.ub)
                score = func(self.positions[i])
                num_evaluations += 1
                
                if score < self.best_scores[i]:
                    self.best_scores[i] = score
                    self.best_positions[i] = self.positions[i].copy()
                    
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i].copy()
                
                phi = np.random.uniform(0, 1)
                self.velocities[i] = (self.w * self.velocities[i] 
                                      + self.c1 * phi * (self.best_positions[i] - self.positions[i]) 
                                      + self.c2 * phi * (self.global_best_position - self.positions[i]))
                
                self.positions[i] += self.velocities[i] + self.levy_flight()
                
            self.w = 0.4 + 0.5 * (num_evaluations / self.budget)

    def __call__(self, func):
        self.optimize(func)
        return self.global_best_position, self.global_best_score