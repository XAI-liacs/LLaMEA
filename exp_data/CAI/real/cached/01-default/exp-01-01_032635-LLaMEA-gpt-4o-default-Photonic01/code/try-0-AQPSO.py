import numpy as np

class AQPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_values = None
        self.global_best_position = None
        self.global_best_value = np.inf
        self.iteration = 0

    def initialize_particles(self, lb, ub):
        self.positions = np.random.uniform(lb, ub, (self.budget, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.budget, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_values = np.full(self.budget, np.inf)

    def quantum_behavior(self, position, personal_best, global_best):
        return position + np.random.uniform(-1, 1, self.dim) * (
            np.abs(global_best - personal_best) / np.linalg.norm(global_best - personal_best + 1e-10)
        )

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_particles(lb, ub)

        while self.iteration < self.budget:
            for i in range(self.budget):
                # Evaluate the current position
                current_value = func(self.positions[i])
                # Update personal best
                if current_value < self.personal_best_values[i]:
                    self.personal_best_values[i] = current_value
                    self.personal_best_positions[i] = self.positions[i].copy()
                # Update global best
                if current_value < self.global_best_value:
                    self.global_best_value = current_value
                    self.global_best_position = self.positions[i].copy()

            # Update velocities and positions
            for i in range(self.budget):
                cognitive = self.quantum_behavior(
                    self.positions[i], self.personal_best_positions[i], self.global_best_position
                )
                self.velocities[i] = cognitive
                self.positions[i] = np.clip(self.positions[i] + self.velocities[i], lb, ub)

            self.iteration += 1

        return self.global_best_position, self.global_best_value