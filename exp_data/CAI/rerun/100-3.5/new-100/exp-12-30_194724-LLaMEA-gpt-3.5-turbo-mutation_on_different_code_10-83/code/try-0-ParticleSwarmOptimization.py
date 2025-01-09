import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, num_particles=30, inertia_weight=0.5, cognitive_weight=1.5, social_weight=1.5):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        particles = np.random.uniform(bounds[0], bounds[1], size=(self.num_particles, self.dim))
        velocities = np.zeros((self.num_particles, self.dim))
        p_best_positions = particles.copy()
        p_best_values = np.full(self.num_particles, np.Inf)
        g_best_position = None
        g_best_value = np.Inf

        for _ in range(self.budget):
            for i in range(self.num_particles):
                f = func(particles[i])
                if f < p_best_values[i]:
                    p_best_values[i] = f
                    p_best_positions[i] = particles[i]
                if f < g_best_value:
                    g_best_value = f
                    g_best_position = particles[i]

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = self.inertia_weight * velocities[i] + \
                                self.cognitive_weight * r1 * (p_best_positions[i] - particles[i]) + \
                                self.social_weight * r2 * (g_best_position - particles[i])
                particles[i] = np.clip(particles[i] + velocities[i], bounds[0], bounds[1])

        self.f_opt = g_best_value
        self.x_opt = g_best_position
        return self.f_opt, self.x_opt