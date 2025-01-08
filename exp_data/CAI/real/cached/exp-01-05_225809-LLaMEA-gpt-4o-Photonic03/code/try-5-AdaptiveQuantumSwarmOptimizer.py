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
        self.alpha = 0.75

    def levy_flight(self, L):
        beta = 1.5
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / abs(v) ** (1 / beta)
        return L * step

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
                step_size = self.levy_flight(0.1)
                mutation = np.random.normal(0, 0.05, self.dim)
                self.position[i] = (self.alpha * r1 * p_best +
                                    (1 - self.alpha) * r2 * global_best) + step_size + mutation
            
            self.alpha = 0.5 + 0.5 * (1 - evaluations / self.budget)
            self.position = np.clip(self.position, lb, ub)

        return self.global_best_position, self.global_best_score