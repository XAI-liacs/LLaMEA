import numpy as np

class QuantumInspiredPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(30, self.budget // (2 * dim))
        self.phi = 0.5 + np.log(self.dim) / np.log(2)
        self.gamma = 0.5
        self.alpha = 0.9  # Adaptive inertia weight
        self.positions = np.random.rand(self.population_size, self.dim)
        self.velocities = np.random.normal(scale=0.1, size=(self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)

    def __call__(self, func):
        global_best_position = None
        global_best_score = np.inf
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                self.positions[i] = np.clip(self.positions[i], func.bounds.lb, func.bounds.ub)

                score = func(self.positions[i])
                evaluations += 1

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = self.positions[i]

                if evaluations >= self.budget:
                    break

            if evaluations < self.budget:
                for i in range(self.population_size):
                    self.velocities[i] = (self.alpha * self.velocities[i] 
                                  + self.phi * (self.personal_best_positions[i] - self.positions[i])
                                  + self.gamma * (global_best_position - self.positions[i]))
                    
                    if evaluations / self.budget > 0.5:  # Multi-phase approach
                        self.gamma *= 1.01  # Increase exploitation
                    else:
                        self.gamma *= 0.99  # Increase exploration
                        
                    self.positions[i] += self.velocities[i]

        return global_best_position, global_best_score