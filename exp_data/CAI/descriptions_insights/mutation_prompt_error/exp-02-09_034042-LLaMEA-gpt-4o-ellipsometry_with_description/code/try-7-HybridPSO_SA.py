import numpy as np

class HybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.inertia_weight = 0.9
        self.cognitive_constant = 1.5
        self.social_constant = 1.5
        self.temperature_initial = 1.0
        self.temperature_final = 0.01
        self.restart_threshold = 10  # New: Restart threshold
        
    def initialize(self, bounds):
        self.positions = np.random.uniform(bounds.lb, bounds.ub, (self.num_particles, self.dim))
        self.velocities = np.zeros((self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, float('inf'))
        self.global_best_position = np.copy(self.personal_best_positions[0])
        self.global_best_score = float('inf')
        self.stagnation_counter = 0  # New: Stagnation counter

    def update_velocity_position(self, bounds):
        r1 = np.random.rand(self.num_particles, self.dim)
        r2 = np.random.rand(self.num_particles, self.dim)
        cognitive_velocity = self.cognitive_constant * r1 * (self.personal_best_positions - self.positions)
        social_velocity = self.social_constant * r2 * (self.global_best_position - self.positions)
        self.velocities = (self.inertia_weight * self.velocities) + cognitive_velocity + social_velocity
        self.positions += self.velocities
        self.positions = np.clip(self.positions, bounds.lb, bounds.ub)
        self.inertia_weight *= 0.98  # Modified: Adjusted inertia weight decay

    def simulated_annealing(self, position, score, bounds):
        temperature = self.temperature_initial
        while temperature > self.temperature_final:
            new_position = position + np.random.normal(0, 0.1, self.dim)
            new_position = np.clip(new_position, bounds.lb, bounds.ub)
            new_score = self.func(new_position)
            if new_score < score or np.random.rand() < np.exp((score - new_score) / temperature):
                position, score = new_position, new_score
            temperature *= 0.9  # Modified: Faster adaptive temperature decay
        return position, score

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
                    self.stagnation_counter = 0  # New: Reset counter on improvement
                else:
                    self.stagnation_counter += 1

                if current_score < self.global_best_score:
                    self.global_best_position = self.positions[i]
                    self.global_best_score = current_score

            self.update_velocity_position(bounds)

            if self.stagnation_counter >= self.restart_threshold:  # New: Check stagnation
                self.initialize(bounds)  # New: Random restart

            self.global_best_position, self.global_best_score = self.simulated_annealing(self.global_best_position, self.global_best_score, bounds)

        return self.global_best_position, self.global_best_score