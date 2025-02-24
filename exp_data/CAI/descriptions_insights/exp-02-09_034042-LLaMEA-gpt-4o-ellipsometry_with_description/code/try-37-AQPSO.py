import numpy as np

class AQPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.alpha = 0.6
        self.beta = 0.4
        self.restart_threshold = 15
        self.stagnation_counter = 0

    def initialize(self, bounds):
        self.positions = np.random.uniform(bounds.lb, bounds.ub, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, float('inf'))
        self.global_best_position = np.copy(self.personal_best_positions[0])
        self.global_best_score = float('inf')

    def update_quantum_velocity_position(self, bounds):
        for i in range(self.num_particles):
            phi = np.random.uniform(0, 1, self.dim)
            self.velocities[i] = self.alpha * np.log(np.abs(1/phi)) * (self.personal_best_positions[i] - self.positions[i]) + \
                                 self.beta * (self.global_best_position - self.positions[i])
            self.positions[i] += self.velocities[i]
            self.positions[i] = np.clip(self.positions[i], bounds.lb, bounds.ub)

    def dynamic_tuning(self, score_improvement):
        if score_improvement:
            self.alpha *= 0.97  # Changed from 0.98 to 0.97
            self.beta *= 1.02   # Changed from 1.01 to 1.02
        else:
            self.alpha *= 1.02  # Changed from 1.01 to 1.02
            self.beta *= 0.98   # Changed from 0.99 to 0.98

    def opposition_based_learning(self, bounds):
        opposition_positions = bounds.lb + bounds.ub - self.positions
        for i in range(self.num_particles):
            if self.func(opposition_positions[i]) < self.personal_best_scores[i]:
                self.positions[i] = opposition_positions[i]

    def __call__(self, func):
        self.func = func
        bounds = func.bounds
        self.initialize(bounds)

        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.num_particles):
                current_score = self.func(self.positions[i])
                evaluations += 1

                if current_score < self.personal_best_scores[i]:
                    self.personal_best_positions[i] = self.positions[i]
                    self.personal_best_scores[i] = current_score
                    self.stagnation_counter = 0
                else:
                    self.stagnation_counter += 1

                if current_score < self.global_best_score:
                    self.global_best_position = self.positions[i]
                    self.global_best_score = current_score

            score_improvement = self.global_best_score < np.min(self.personal_best_scores)
            self.dynamic_tuning(score_improvement)
            self.update_quantum_velocity_position(bounds)

            if self.stagnation_counter >= self.restart_threshold:
                self.opposition_based_learning(bounds)  # Add opposition-based learning
                self.initialize(bounds)

        return self.global_best_position, self.global_best_score