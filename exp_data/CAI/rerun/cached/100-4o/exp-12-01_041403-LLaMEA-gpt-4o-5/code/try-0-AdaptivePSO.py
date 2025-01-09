import numpy as np

class AdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 + 2 * int(np.sqrt(self.dim))
        self.c1 = 1.49445  # cognitive coefficient
        self.c2 = 1.49445  # social coefficient
        self.inertia_max = 0.9
        self.inertia_min = 0.4
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound,
                                           (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evals = 0

    def __call__(self, func):
        while self.evals < self.budget:
            for i in range(self.population_size):
                if self.evals >= self.budget:
                    break

                current_score = func(self.positions[i])
                self.evals += 1

                if current_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = current_score
                    self.personal_best_positions[i] = self.positions[i]

                if current_score < self.global_best_score:
                    self.global_best_score = current_score
                    self.global_best_position = self.positions[i]

            inertia_weight = self.inertia_max - (self.inertia_max - self.inertia_min) * (self.evals / self.budget)

            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = (inertia_weight * self.velocities[i] + cognitive_component + social_component)

                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

        return self.global_best_position, self.global_best_score