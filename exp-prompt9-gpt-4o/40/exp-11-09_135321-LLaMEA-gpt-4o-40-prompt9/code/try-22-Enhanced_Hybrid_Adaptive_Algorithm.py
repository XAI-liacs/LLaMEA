import numpy as np
import random

class Enhanced_Hybrid_Adaptive_Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

        # Enhanced Parameters
        self.num_particles = 50
        self.inertia_weight = 0.5
        self.cognitive_coeff = 1.4
        self.social_coeff = 1.8
        self.learning_rate = 0.6
        
        # Chaotic Initialization
        self.positions = self.chaotic_initialization(self.num_particles, self.dim)
        self.velocities = np.random.uniform(-1.0, 1.0, (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, np.inf)

        self.global_best_position = None
        self.global_best_score = np.inf

    def chaotic_initialization(self, num_particles, dim):
        # Logistic map-based initialization for diversity
        x0 = random.random()
        positions = np.zeros((num_particles, dim))
        for i in range(num_particles):
            for j in range(dim):
                x0 = 4.0 * x0 * (1.0 - x0)
                positions[i][j] = self.lower_bound + (self.upper_bound - self.lower_bound) * x0
        return positions

    def __call__(self, func):
        evals = 0
        while evals < self.budget:
            scores = np.apply_along_axis(func, 1, self.positions)
            evals += self.num_particles

            for i in range(self.num_particles):
                if scores[i] < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = scores[i]
                    self.personal_best_positions[i] = self.positions[i]

                if scores[i] < self.global_best_score:
                    self.global_best_score = scores[i]
                    self.global_best_position = self.positions[i]

            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions - self.positions)
            social_component = self.social_coeff * r2 * (self.global_best_position - self.positions)
            self.velocities = self.inertia_weight * self.velocities + cognitive_component + social_component
            self.positions += self.learning_rate * self.velocities
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            # Adaptive Differential Evolution
            F = 0.5 + np.random.rand() * 0.5
            CR = 0.8 + np.random.rand() * 0.2

            for i in range(self.num_particles):
                indices = [idx for idx in range(self.num_particles) if idx != i]
                x1, x2, x3 = self.positions[np.random.choice(indices, 3, replace=False)]
                mutant_vector = np.clip(x1 + F * (x2 - x3), self.lower_bound, self.upper_bound)
                trial_vector = np.where(np.random.rand(self.dim) < CR, mutant_vector, self.positions[i])

                trial_score = func(trial_vector)

                if trial_score < scores[i]:
                    self.positions[i] = trial_vector
                    scores[i] = trial_score

            evals += self.num_particles

        return self.global_best_position, self.global_best_score