import numpy as np

class EnhancedQuantumPSO:
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
        self.local_search_prob = 0.1  # probability to perform local search

    def adaptive_local_search(self, position, func):
        # Perform a small local search around a given position
        step_size = 0.01 * (func.bounds.ub - func.bounds.lb)
        candidate = position + np.random.uniform(-step_size, step_size)
        candidate = np.clip(candidate, func.bounds.lb, func.bounds.ub)
        return candidate, func(candidate)

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
                
                self.positions[i] += self.velocities[i]
                
                if np.random.rand() < self.local_search_prob:
                    candidate_position, candidate_score = self.adaptive_local_search(self.positions[i], func)
                    num_evaluations += 1
                    if candidate_score < self.best_scores[i]:
                        self.best_scores[i] = candidate_score
                        self.best_positions[i] = candidate_position
                        
                    if candidate_score < self.global_best_score:
                        self.global_best_score = candidate_score
                        self.global_best_position = candidate_position

            self.w = np.random.uniform(0.4, 0.9)  # Stochastic inertia adaptation
            self.local_search_prob = 0.1 + 0.4 * (1 - num_evaluations / self.budget)

    def __call__(self, func):
        self.optimize(func)
        return self.global_best_position, self.global_best_score