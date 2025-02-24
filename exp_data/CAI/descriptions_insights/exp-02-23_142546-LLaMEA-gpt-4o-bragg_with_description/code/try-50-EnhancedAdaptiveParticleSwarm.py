import numpy as np

class EnhancedAdaptiveParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.max_inertia_weight = 0.9
        self.min_inertia_weight = 0.4
        self.base_cognitive_param = 1.5
        self.base_social_param = 1.5
        self.best_global_position = None
        self.best_global_value = float('-inf')
        self.max_velocity = None

    def chaotic_map(self, x):
        return 4 * x * (1 - x)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.max_velocity = 0.1 * (ub - lb)
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-self.max_velocity, self.max_velocity, (self.num_particles, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.full(self.num_particles, float('-inf'))

        chaotic_factor = np.random.rand()
        evaluations = 0
        while evaluations < self.budget:
            inertia_weight = self.max_inertia_weight - (self.max_inertia_weight - self.min_inertia_weight) * (evaluations / self.budget)
            for i in range(self.num_particles):
                value = func(particles[i])
                evaluations += 1

                if value > personal_best_values[i]:
                    personal_best_values[i] = value
                    personal_best_positions[i] = particles[i]

                if value > self.best_global_value:
                    self.best_global_value = value
                    self.best_global_position = particles[i]

            adaptive_factor = evaluations / self.budget
            cognitive_param = self.base_cognitive_param + 0.5 * np.sin(2 * np.pi * adaptive_factor)
            social_param = self.base_social_param - 0.5 * np.sin(2 * np.pi * adaptive_factor)

            for i in range(self.num_particles):
                r1, r2 = np.random.random(2)
                chaotic_factor = self.chaotic_map(chaotic_factor) # Use chaotic map
                cognitive_velocity = cognitive_param * r1 * (personal_best_positions[i] - particles[i])
                social_velocity = social_param * r2 * (self.best_global_position - particles[i])
                velocities[i] = inertia_weight * velocities[i] + chaotic_factor * (cognitive_velocity + social_velocity)
                velocities[i] = np.clip(velocities[i], -self.max_velocity, self.max_velocity)
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)

                # Adaptive learning rate for Lévy flight
                if evaluations > self.budget * 0.7:
                    learning_rate = 0.01 * (1 - evaluations / self.budget) 
                    particles[i] += learning_rate * self.levy_flight(self.dim)

                # Dynamic restart mechanism to maintain exploration 
                if np.random.rand() < 0.05 + 0.1 * (1 - evaluations / self.budget):
                    particles[i] = np.random.uniform(lb, ub, self.dim)

        return self.best_global_position