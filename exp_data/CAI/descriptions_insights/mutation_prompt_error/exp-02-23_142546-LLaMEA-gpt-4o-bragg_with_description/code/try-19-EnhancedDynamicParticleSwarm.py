import numpy as np

class EnhancedDynamicParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_num_particles = 30
        self.num_particles = self.initial_num_particles
        self.max_inertia_weight = 0.9
        self.min_inertia_weight = 0.4
        self.cognitive_param = 1.5
        self.social_param = 1.5
        self.best_global_position = None
        self.best_global_value = float('-inf')
        self.max_velocity = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.max_velocity = 0.1 * (ub - lb)
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-self.max_velocity, self.max_velocity, (self.num_particles, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.full(self.num_particles, float('-inf'))

        evaluations = 0
        while evaluations < self.budget:
            inertia_weight = self.max_inertia_weight - (self.max_inertia_weight - self.min_inertia_weight) * (evaluations / self.budget)
            adaptive_factor = evaluations / self.budget

            if evaluations % (self.budget // 10) == 0 and self.num_particles < 60:
                additional_particles = 5
                new_particles = np.random.uniform(lb, ub, (additional_particles, self.dim))
                new_velocities = np.random.uniform(-self.max_velocity, self.max_velocity, (additional_particles, self.dim))
                particles = np.vstack((particles, new_particles))
                velocities = np.vstack((velocities, new_velocities))
                personal_best_positions = np.vstack((personal_best_positions, new_particles))
                personal_best_values = np.concatenate((personal_best_values, np.full(additional_particles, float('-inf'))))
                self.num_particles += additional_particles

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
                cognitive_velocity = (self.cognitive_param + adaptive_factor) * r1 * (personal_best_positions[i] - particles[i])
                social_velocity = (self.social_param - adaptive_factor) * r2 * (self.best_global_position - particles[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_velocity + social_velocity
                velocities[i] = np.clip(velocities[i], -self.max_velocity, self.max_velocity)
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)
                
                if np.random.rand() < 0.05:  
                    particles[i] = np.random.uniform(lb, ub, self.dim)

        return self.best_global_position