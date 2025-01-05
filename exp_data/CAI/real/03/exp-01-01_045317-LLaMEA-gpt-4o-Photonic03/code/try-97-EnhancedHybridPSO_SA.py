import numpy as np

class EnhancedHybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.inertia_weight = 0.9
        self.cognitive_const = 1.7  # Adjusted cognitive constant for stronger personal attraction
        self.social_const = 1.7  # Adjusted social constant for stronger global attraction
        self.temperature = 1.0
        self.cooling_rate = 0.993  # Slight change in cooling rate
        self.best_global_position = None
        self.best_global_value = np.inf

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-0.5, 0.5, (self.num_particles, self.dim))  # Reduced initial velocity range
        personal_best_positions = np.copy(particles)
        personal_best_values = np.array([func(p) for p in particles])
        evaluations = self.num_particles

        self.best_global_position = particles[np.argmin(personal_best_values)]
        self.best_global_value = np.min(personal_best_values)

        while evaluations < self.budget:
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(2, self.dim)
                self.inertia_weight = 0.4 + 0.5 * (self.budget - evaluations) / self.budget  # Adaptive inertia weight
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_const * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.social_const * r2 * (self.best_global_position - particles[i]))

                if evaluations % 20 == 0:  # Less frequent, but stronger perturbation
                    velocities[i] += np.random.randn(self.dim) * 0.3

                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)

                current_value = func(particles[i])
                evaluations += 1

                if current_value < personal_best_values[i]:
                    personal_best_values[i] = current_value
                    personal_best_positions[i] = particles[i]

                acceptance_probability = np.exp((self.best_global_value - current_value) / self.temperature)
                if current_value < self.best_global_value or acceptance_probability > np.random.rand():
                    self.best_global_position = particles[i]
                    self.best_global_value = current_value

            self.temperature *= (0.8 + 0.2 * np.cos(evaluations / self.budget * np.pi))  # Non-linear cooling with adjusted parameters

            if evaluations % (self.num_particles * 5) == 0:
                perturbation_factor = 0.05 * (1 - evaluations / self.budget)
                particles += np.random.uniform(-perturbation_factor, perturbation_factor, (self.num_particles, self.dim))

        return self.best_global_position, self.best_global_value