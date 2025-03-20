import numpy as np

class HybridPSOSimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.temperature = 100.0
        self.cooling_rate = 0.9  # Modified for more gradual cooling
        self.w = 0.8  # Reduced to improve balance between exploration and exploitation
        self.c1 = 1.7  # Increased to enhance personal best influence
        self.c2 = 1.3  # Decreased to reduce global best influence

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        X = np.random.uniform(lb, ub, (self.population_size, self.dim))
        V = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = X.copy()
        personal_best_values = np.array([func(x) for x in X])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)
        
        evaluations = self.population_size

        while evaluations < self.budget:
            r1, r2 = np.random.rand(2)
            V = self.w * V + \
                self.c1 * r1 * (personal_best - X) + \
                self.c2 * r2 * (global_best - X)
            X = np.clip(X + V, lb, ub)

            # Simulated Annealing step
            for i in range(self.population_size):
                candidate = X[i] + np.random.uniform(-0.5, 0.5, self.dim) * (ub - lb) * self.temperature
                candidate = np.clip(candidate, lb, ub)
                candidate_value = func(candidate)
                evaluations += 1

                if candidate_value < personal_best_values[i]:
                    personal_best[i] = candidate
                    personal_best_values[i] = candidate_value

                    if candidate_value < global_best_value:
                        global_best = candidate
                        global_best_value = candidate_value
                elif np.exp((personal_best_values[i] - candidate_value) / self.temperature) > np.random.rand():
                    personal_best[i] = candidate
                    personal_best_values[i] = candidate_value

            if evaluations >= self.budget:
                break

            self.temperature *= self.cooling_rate

        return global_best