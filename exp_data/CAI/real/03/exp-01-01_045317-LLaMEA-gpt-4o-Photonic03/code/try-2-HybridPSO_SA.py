import numpy as np

class HybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.inertia_weight = 0.9  # Changed this line: increased initial inertia weight
        self.cognitive_const = 1.5
        self.social_const = 1.5
        self.temperature = 1.0
        self.cooling_rate = 0.98  # Changed this line: slightly increased cooling rate
        self.best_global_position = None
        self.best_global_value = np.inf

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.array([func(p) for p in particles])
        evaluations = self.num_particles

        self.best_global_position = particles[np.argmin(personal_best_values)]
        self.best_global_value = np.min(personal_best_values)

        while evaluations < self.budget:
            for i in range(self.num_particles):
                # Update velocities
                r1, r2 = np.random.rand(2, self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_const * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.social_const * r2 * (self.best_global_position - particles[i]))

                # Update positions
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)

                # Evaluate particle
                current_value = func(particles[i])
                evaluations += 1

                # Update personal best
                if current_value < personal_best_values[i]:
                    personal_best_values[i] = current_value
                    personal_best_positions[i] = particles[i]

                # Simulated Annealing acceptance criterion
                if current_value < self.best_global_value or np.exp((self.best_global_value - current_value) / self.temperature) > np.random.rand():
                    self.best_global_position = particles[i]
                    self.best_global_value = current_value

            # Cool the temperature
            self.temperature *= self.cooling_rate

        return self.best_global_position, self.best_global_value