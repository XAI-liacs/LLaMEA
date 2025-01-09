import numpy as np

class AdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(40, budget // 2)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, dim))
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, float('inf'))
        self.global_best_position = np.zeros(dim)
        self.global_best_score = float('inf')
        self.c1 = 2.0  # cognitive component
        self.c2 = 2.0  # social component
        self.w_start = 0.9  # starting inertia weight
        self.w_end = 0.4  # ending inertia weight

    def __call__(self, func):
        eval_count = 0

        while eval_count < self.budget:
            for i in range(self.num_particles):
                score = func(self.positions[i])
                eval_count += 1

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]

                if eval_count >= self.budget:
                    break

            w = self.w_start - ((self.w_start - self.w_end) * eval_count / self.budget)

            for i in range(self.num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_velocity = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_velocity = self.c2 * r2 * (self.global_best_position - self.positions[i])

                self.velocities[i] = w * self.velocities[i] + cognitive_velocity + social_velocity
                self.positions[i] += self.velocities[i] + 0.5 * (np.random.rand(self.dim) - 0.5) * (self.upper_bound - self.lower_bound) * 0.1

                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

        return self.global_best_position, self.global_best_score