import numpy as np

class AdaptiveQuantumSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.position = np.random.rand(self.population_size, dim)
        self.velocity = np.random.rand(self.population_size, dim) * (0.15 + 0.05 * np.random.rand())  # Modified line
        self.best_positions = np.copy(self.position)
        self.best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.alpha = 0.75

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.position = lb + (ub - lb) * self.position
        evaluations = 0
        
        while evaluations < self.budget:
            elite_idx = np.argmin(self.best_scores)  # Track elite solution
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
            self.population_size = 10 + int((1 - evaluations / self.budget) * 2 * np.sqrt(self.dim))  # New line
            for i in range(self.population_size):
                p_best = self.best_positions[i]
                global_best = self.global_best_position
                decay_factor = 0.5 * (1 - evaluations / self.budget)  # Modified line
                diversity = np.std(self.position, axis=0).mean()  # New line
                weight = 0.1 + 0.9 * (1 - evaluations / self.budget)  # Modified line
                turbulence = np.random.normal(0, weight * 0.05 * (1 + diversity) * (1 - evaluations / self.budget), self.dim)  # Modified line
                levy = np.random.normal(0, 1, self.dim) * decay_factor * 0.5 * (self.best_positions[elite_idx] - self.position[i])  # Modified line
                inertia = 0.5 + 0.4 * (1 - evaluations / self.budget)  # Changed line
                self.velocity[i] = inertia * self.velocity[i] + (self.alpha * r1 * (p_best - self.position[i]) + (1 - self.alpha) * r2 * (global_best - self.position[i])) + levy + turbulence  # Modified line
                self.position[i] += self.velocity[i]
            
            self.alpha = 0.5 + 0.5 * (1 - evaluations / self.budget)

            self.position = np.clip(self.position, lb, ub)

        return self.global_best_position, self.global_best_score