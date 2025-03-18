import numpy as np

class EnhancedHybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.4
        self.cognitive_constant = 1.5
        self.social_constant = 1.5
        self.temperature_initial = 1.0
        self.temperature_final = 0.01

    def initialize(self, bounds):
        self.positions = np.random.uniform(bounds.lb, bounds.ub, (self.num_particles, self.dim))
        self.velocities = np.zeros((self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, float('inf'))
        self.global_best_position = np.copy(self.personal_best_positions[0])
        self.global_best_score = float('inf')

    def update_velocity_position(self, bounds, iter_count, max_iter):
        inertia_weight = self.inertia_weight_initial - (
            (self.inertia_weight_initial - self.inertia_weight_final) * (iter_count / max_iter))
        r1 = np.random.rand(self.num_particles, self.dim)
        r2 = np.random.rand(self.num_particles, self.dim)
        cognitive_velocity = self.cognitive_constant * r1 * (self.personal_best_positions - self.positions)
        social_velocity = self.social_constant * r2 * (self.global_best_position - self.positions)
        self.velocities = (inertia_weight * self.velocities) + cognitive_velocity + social_velocity
        self.positions += self.velocities
        self.positions = np.clip(self.positions, bounds.lb, bounds.ub)

    def adaptive_simulated_annealing(self, position, score, bounds, iter_count, max_iter):
        temperature = self.temperature_initial * (1 - iter_count / max_iter)
        new_position = position + np.random.normal(0, 0.1, self.dim)
        new_position = np.clip(new_position, bounds.lb, bounds.ub)
        new_score = self.func(new_position)
        if new_score < score or np.random.rand() < np.exp((score - new_score) / temperature):
            return new_position, new_score
        return position, score

    def __call__(self, func):
        self.func = func
        bounds = func.bounds
        self.initialize(bounds)

        evaluations = 0
        iter_count = 0
        max_iter = self.budget // self.num_particles

        while evaluations < self.budget:
            for i in range(self.num_particles):
                current_score = self.func(self.positions[i])
                evaluations += 1

                if current_score < self.personal_best_scores[i]:
                    self.personal_best_positions[i] = self.positions[i]
                    self.personal_best_scores[i] = current_score

                if current_score < self.global_best_score:
                    self.global_best_position = self.positions[i]
                    self.global_best_score = current_score

            self.update_velocity_position(bounds, iter_count, max_iter)

            # Apply adaptive simulated annealing on global best position
            self.global_best_position, self.global_best_score = self.adaptive_simulated_annealing(
                self.global_best_position, self.global_best_score, bounds, iter_count, max_iter)

            iter_count += 1

        return self.global_best_position, self.global_best_score