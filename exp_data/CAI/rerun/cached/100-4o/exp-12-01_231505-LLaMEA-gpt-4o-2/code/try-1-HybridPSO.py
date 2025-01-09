import numpy as np

class HybridPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(100, int(budget / 10))
        self.inertia = 0.7
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5
        self.mutation_prob = 0.1
        self.vel_max = (self.upper_bound - self.lower_bound) * 0.1

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-self.vel_max, self.vel_max, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(pos) for pos in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = self.population_size

        while evaluations < self.budget:
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (self.inertia * velocities +
                          self.cognitive_coefficient * r1 * (personal_best_positions - positions) +
                          self.social_coefficient * r2 * (global_best_position - positions))
            velocities = np.clip(velocities, -self.vel_max, self.vel_max)
            positions += velocities
            positions = np.clip(positions, self.lower_bound, self.upper_bound)

            for i in range(self.population_size):
                if np.random.rand() < self.mutation_prob:
                    mutation_strength = 0.1 * (self.upper_bound - self.lower_bound)
                    mutation = np.random.normal(0, mutation_strength, self.dim)
                    positions[i] = np.clip(positions[i] + mutation, self.lower_bound, self.upper_bound)

            scores = np.array([func(pos) for pos in positions])
            evaluations += self.population_size

            for i in range(self.population_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = positions[i]

            if np.min(scores) < global_best_score:
                global_best_score = np.min(scores)
                global_best_position = positions[np.argmin(scores)]

        return global_best_position