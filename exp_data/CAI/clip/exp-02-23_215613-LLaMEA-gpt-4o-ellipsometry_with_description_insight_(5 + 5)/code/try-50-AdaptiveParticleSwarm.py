import numpy as np

class AdaptiveParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        num_particles = 10
        positions = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_particles, self.dim))
        velocities = np.random.uniform(-0.1, 0.1, (num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_values = np.array([func(pos) for pos in positions])
        global_best_idx = np.argmin(personal_best_values)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_value = personal_best_values[global_best_idx]

        while self.evaluations < self.budget:
            for i in range(num_particles):
                velocities[i] += np.random.rand(self.dim) * (personal_best_positions[i] - positions[i]) + \
                                 np.random.rand(self.dim) * (global_best_position - positions[i])
                positions[i] += velocities[i]
                
                positions[i] = np.clip(positions[i], bounds[:, 0], bounds[:, 1])
                current_value = func(positions[i])
                self.evaluations += 1

                if current_value < personal_best_values[i]:
                    personal_best_values[i] = current_value
                    personal_best_positions[i] = positions[i]

                if current_value < global_best_value:
                    global_best_value = current_value
                    global_best_position = positions[i]

                if self.evaluations >= self.budget:
                    break

            # Dynamic neighborhood adjustment
            inertia_weight = 0.5 + np.random.rand() / 2.0
            velocities *= inertia_weight

        return global_best_position