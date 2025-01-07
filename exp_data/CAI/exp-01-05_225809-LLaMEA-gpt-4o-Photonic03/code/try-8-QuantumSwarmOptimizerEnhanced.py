import numpy as np

class QuantumSwarmOptimizerEnhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.position = np.random.rand(self.population_size, dim)
        self.best_positions = np.copy(self.position)
        self.best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.alpha = 0.75  # Initial exploration-exploitation balance parameter
        self.beta = 0.5  # New parameter for adaptive convergence strategy

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.position = lb + (ub - lb) * self.position
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                score = func(self.position[i])
                evaluations += 1

                if score < self.best_scores[i]:
                    self.best_scores[i] = score
                    self.best_positions[i] = self.position[i]

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.position[i]

            r1, r2 = np.random.rand(2)
            for i in range(self.population_size):
                p_best = self.best_positions[i]
                global_best = self.global_best_position
                self.position[i] = (self.alpha * r1 * p_best +
                                    self.beta * r2 * global_best) + np.random.normal(0, 0.1, self.dim)
            
            self.alpha = 0.5 + 0.5 * (1 - evaluations / self.budget)
            self.beta = 0.3 + 0.7 * (evaluations / self.budget)  # Adaptive beta for exploitation

            self.position = np.clip(self.position, lb, ub)

        return self.global_best_position, self.global_best_score