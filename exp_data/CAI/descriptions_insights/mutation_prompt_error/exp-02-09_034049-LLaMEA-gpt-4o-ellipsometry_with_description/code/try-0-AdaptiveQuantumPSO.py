import numpy as np

class AdaptiveQuantumPSO:
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
        
    def optimize(self, func):
        num_evaluations = 0
        
        while num_evaluations < self.budget:
            for i in range(self.swarm_size):
                # Ensure positions are within bounds
                self.positions[i] = np.clip(self.positions[i], func.bounds.lb, func.bounds.ub)
                
                # Evaluate the function
                score = func(self.positions[i])
                num_evaluations += 1
                
                # Update personal best
                if score < self.best_scores[i]:
                    self.best_scores[i] = score
                    self.best_positions[i] = self.positions[i].copy()
                    
                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i].copy()
                
                # Quantum-inspired update for position
                phi = np.random.uniform(0, 1)
                self.velocities[i] = (self.w * self.velocities[i] 
                                      + self.c1 * phi * (self.best_positions[i] - self.positions[i]) 
                                      + self.c2 * phi * (self.global_best_position - self.positions[i]))
                
                self.positions[i] += self.velocities[i]
                
            # Dynamic adaptation of parameters
            self.w = 0.4 + 0.5 * (num_evaluations / self.budget)  # decreasing inertia over time

    def __call__(self, func):
        self.optimize(func)
        return self.global_best_position, self.global_best_score