import numpy as np

class AdaptiveQuantumSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.position = np.random.rand(self.population_size, dim)
        self.best_positions = np.copy(self.position)
        self.best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.alpha = 0.75  # Exploration-Exploitation balance parameter

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.position = lb + (ub - lb) * self.position
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                # Evaluate current position
                score = func(self.position[i])
                evaluations += 1

                # Update best position of swarm
                if score < self.best_scores[i]:
                    self.best_scores[i] = score
                    self.best_positions[i] = self.position[i]

                # Update global best position
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.position[i]

            # Update positions with quantum-inspired approach
            r1, r2 = np.random.rand(2)
            for i in range(self.population_size):
                p_best = self.best_positions[i]
                global_best = self.global_best_position
                self.alpha = 0.9 - 0.5 * (evaluations / self.budget)  # Adjust alpha dynamically
                self.position[i] = (self.alpha * r1 * p_best +
                                    (1 - self.alpha) * r2 * global_best) + np.random.normal(0, 0.1, self.dim)

            # Ensure position stays within bounds
            self.position = np.clip(self.position, lb, ub)

        return self.global_best_position, self.global_best_score