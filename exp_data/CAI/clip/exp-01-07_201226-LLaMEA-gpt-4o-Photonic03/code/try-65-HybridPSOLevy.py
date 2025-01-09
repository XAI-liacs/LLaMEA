import numpy as np

class HybridPSOLevy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.iterations = budget // dim
        self.population_size = 30
        self.w = 0.5  # inertia weight
        self.c1 = 2.05  # cognitive coefficient
        self.c2 = 2.05  # social coefficient
        self.levy_scale = 0.1  # scale for Lévy flight

    def levy_flight(self, size):
        beta = 1.5
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=size)
        v = np.random.normal(0, 1, size=size)
        step = u / abs(v) ** (1 / beta)
        return step * self.levy_scale

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(x) for x in personal_best_positions])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        global_best_score = personal_best_scores[global_best_index]
        evals = self.population_size

        for iter_num in range(self.iterations):
            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - positions) +
                          self.c2 * r2 * (global_best_position - positions))
            positions = np.clip(positions + velocities, lb, ub)
            
            # Apply Lévy flight mutation to enhance exploration
            mutation_mask = np.random.rand(self.population_size, self.dim) < 0.1
            levy_steps = self.levy_flight((self.population_size, self.dim))
            positions += mutation_mask * levy_steps
            positions = np.clip(positions, lb, ub)

            scores = np.array([func(x) for x in positions])
            evals += self.population_size

            for i in range(self.population_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_scores[i] = scores[i]
                    if scores[i] < global_best_score:
                        global_best_position = positions[i]
                        global_best_score = scores[i]

            if evals >= self.budget:
                break

        return global_best_position, global_best_score