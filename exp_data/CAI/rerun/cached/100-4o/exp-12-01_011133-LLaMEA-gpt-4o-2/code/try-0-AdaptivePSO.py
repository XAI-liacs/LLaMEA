import numpy as np

class AdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = 30
        self.inertia_weight = 0.9
        self.c1 = 2.0  # cognitive coefficient
        self.c2 = 2.0  # social coefficient
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        self.best_positions = np.copy(self.positions)
        self.best_scores = np.full(self.num_particles, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf

    def update_velocities(self):
        for i in range(self.num_particles):
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            cognitive_velocity = self.c1 * r1 * (self.best_positions[i] - self.positions[i])
            social_velocity = self.c2 * r2 * (self.global_best_position - self.positions[i])
            self.velocities[i] = (self.inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity)

    def update_positions(self):
        self.positions += self.velocities
        self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            scores = np.apply_along_axis(func, 1, self.positions)
            evaluations += self.num_particles
            
            # Update personal bests
            better_mask = scores < self.best_scores
            self.best_scores[better_mask] = scores[better_mask]
            self.best_positions[better_mask] = self.positions[better_mask]
            
            # Update global best
            min_score = np.min(scores)
            if min_score < self.global_best_score:
                self.global_best_score = min_score
                self.global_best_position = self.positions[np.argmin(scores)]
            
            # Update inertia weight dynamically
            self.inertia_weight = 0.4 + (0.5 * (self.budget - evaluations) / self.budget)
            
            # Update velocities and positions
            self.update_velocities()
            self.update_positions()

        return self.global_best_position, self.global_best_score