import numpy as np

class AdaptiveMemoryPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initialize_particles()

    def initialize_particles(self):
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf

    def update_velocities(self, w, c1, c2):
        r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
        cognitive_component = c1 * r1 * (self.personal_best_positions - self.positions)
        social_component = c2 * r2 * (self.global_best_position - self.positions)
        self.velocities = w * self.velocities + cognitive_component + social_component

    def update_positions(self):
        self.positions += self.velocities
        np.clip(self.positions, self.lower_bound, self.upper_bound, out=self.positions)

    def evaluate_particles(self, func):
        scores = np.apply_along_axis(func, 1, self.positions)
        better_mask = scores < self.personal_best_scores
        self.personal_best_positions[better_mask] = self.positions[better_mask]
        self.personal_best_scores[better_mask] = scores[better_mask]
        min_score_index = np.argmin(scores)
        if scores[min_score_index] < self.global_best_score:
            self.global_best_score = scores[min_score_index]
            self.global_best_position = self.positions[min_score_index]

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            w = 0.9 - 0.5 * (np.sin((evaluations/self.budget) * np.pi))  # Nonlinear inertia weight
            c1 = 1.5 + (2.0 - 1.5) * (evaluations / self.budget)         # Adaptive cognitive coefficient
            c2 = 2.5 - (2.5 - 1.5) * (evaluations / self.budget)         # Adaptive social coefficient
            self.update_velocities(w, c1, c2)
            self.update_positions()
            self.evaluate_particles(func)
            evaluations += self.num_particles
        return self.global_best_position