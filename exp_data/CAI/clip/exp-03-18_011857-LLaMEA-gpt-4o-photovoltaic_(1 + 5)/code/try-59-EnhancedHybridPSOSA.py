import numpy as np

class EnhancedHybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.initial_temperature = 100.0
        self.cooling_rate = 0.97  # More gradual cooling
        self.initial_w = 0.9  # Start with higher inertia weight
        self.max_w = 0.85  # Reduced maximum inertia weight
        self.min_w = 0.4
        self.c1 = 2.0  # Enhanced to increase personal best attraction
        self.c2 = 1.0  # Reduced to decrease global best attraction

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        X = np.random.uniform(lb, ub, (self.population_size, self.dim))
        V = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = X.copy()
        personal_best_values = np.array([func(x) for x in X])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)
        
        evaluations = self.population_size
        temperature = self.initial_temperature
        w = self.initial_w

        while evaluations < self.budget:
            r1, r2 = np.random.rand(2)
            progress_ratio = evaluations / self.budget
            w = self.max_w - progress_ratio * (self.max_w - self.min_w)
            
            V = w * V + self.c1 * r1 * (personal_best - X) + self.c2 * r2 * (global_best - X)
            X = np.clip(X + V, lb, ub)

            # Adaptive mutation strength
            mutation_strength = 0.5 * (1 - progress_ratio)

            for i in range(self.population_size):
                candidate = X[i] + np.random.uniform(-mutation_strength, mutation_strength, self.dim) * (ub - lb)
                candidate = np.clip(candidate, lb, ub)
                candidate_value = func(candidate)
                evaluations += 1

                if candidate_value < personal_best_values[i]:
                    personal_best[i] = candidate
                    personal_best_values[i] = candidate_value

                    if candidate_value < global_best_value:
                        global_best = candidate
                        global_best_value = candidate_value
                elif np.exp((personal_best_values[i] - candidate_value) / temperature) > np.random.rand():
                    personal_best[i] = candidate
                    personal_best_values[i] = candidate_value

            if evaluations >= self.budget:
                break

            temperature *= self.cooling_rate

        return global_best