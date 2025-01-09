import numpy as np

class AdaptiveNeighborhoodPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = 30
        self.c1 = 2.0
        self.c2 = 2.0
        self.w_max = 0.9
        self.w_min = 0.4
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        self.personal_best = self.particles.copy()
        self.personal_best_value = np.full(self.num_particles, np.inf)
        self.global_best = None
        self.global_best_value = np.inf
        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            for i in range(self.num_particles):
                if self.evaluations >= self.budget:
                    break
                value = func(self.particles[i])
                self.evaluations += 1

                if value < self.personal_best_value[i]:
                    self.personal_best_value[i] = value
                    self.personal_best[i] = self.particles[i].copy()

                if value < self.global_best_value:
                    self.global_best_value = value
                    self.global_best = self.particles[i].copy()

            if self.global_best is not None:
                neighborhood_range = np.clip(self.global_best_value, 1e-3, 5.0) / 5.0
                neighbors = self.particles + np.random.uniform(-neighborhood_range, neighborhood_range, self.particles.shape)
                neighbors = np.clip(neighbors, self.lower_bound, self.upper_bound)

                for i in range(self.num_particles):
                    if self.evaluations >= self.budget:
                        break
                    value = func(neighbors[i])
                    self.evaluations += 1

                    if value < self.personal_best_value[i]:
                        self.personal_best_value[i] = value
                        self.personal_best[i] = neighbors[i].copy()

                    if value < self.global_best_value:
                        self.global_best_value = value
                        self.global_best = neighbors[i].copy()

            w = self.w_max - (self.w_max - self.w_min) * (self.evaluations / self.budget)

            for i in range(self.num_particles):
                cognitive_component = self.c1 * np.random.rand(self.dim) * (self.personal_best[i] - self.particles[i])
                social_component = self.c2 * np.random.rand(self.dim) * (self.global_best - self.particles[i])
                self.velocities[i] = w * self.velocities[i] + cognitive_component + social_component
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

        return self.global_best