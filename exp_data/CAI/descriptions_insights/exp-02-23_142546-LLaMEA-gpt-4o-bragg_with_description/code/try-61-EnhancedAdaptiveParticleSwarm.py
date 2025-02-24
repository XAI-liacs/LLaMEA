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

    def levy_flight(self, L, adaptive_factor):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=L)
        v = np.random.normal(0, 1, size=L)
        return adaptive_factor * (u / np.abs(v) ** (1 / beta))

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
            cognitive_param = self.base_cognitive_param + 0.5 * np.sin(3 * np.pi * adaptive_factor)
            social_param = self.base_social_param - 0.5 * np.sin(3 * np.pi * adaptive_factor)

            for i in range(self.num_particles):
                r1, r2 = np.random.random(2)
                cognitive_velocity = cognitive_param * r1 * (personal_best_positions[i] - particles[i])
                social_velocity = social_param * r2 * (self.best_global_position - particles[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_velocity + social_velocity
                velocities[i] = np.clip(velocities[i], -self.max_velocity, self.max_velocity)
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)

                if evaluations > self.budget * 0.7:
                    particles[i] += 0.01 * self.levy_flight(self.dim, adaptive_factor)

                if np.random.rand() < 0.05 + 0.1 * np.exp(-5 * (evaluations / self.budget)):
                    particles[i] = np.random.uniform(lb, ub, self.dim)

        return self.best_global_position