import numpy as np

class DynamicParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.inertia_weight = 0.7
        self.cognitive_param = 1.5
        self.social_param = 1.5
        self.best_global_position = None
        self.best_global_value = float('-inf')

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.full(self.num_particles, float('-inf'))

        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.num_particles):
                value = func(particles[i])
                evaluations += 1

                if value > personal_best_values[i]:
                    personal_best_values[i] = value
                    personal_best_positions[i] = particles[i]

                if value > self.best_global_value:
                    self.best_global_value = value
                    self.best_global_position = particles[i]

            for i in range(self.num_particles):
                r1, r2 = np.random.random(2)
                cognitive_velocity = self.cognitive_param * r1 * (personal_best_positions[i] - particles[i])
                social_velocity = self.social_param * r2 * (self.best_global_position - particles[i])
                velocities[i] = (self.inertia_weight * velocities[i] 
                                 + cognitive_velocity 
                                 + social_velocity)
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)

        return self.best_global_position