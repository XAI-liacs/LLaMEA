import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, num_particles=30):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.f_opt = np.Inf
        self.x_opt = None
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 2.05
        self.c2 = 2.05

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        p_best_positions = particles.copy()
        p_best_values = np.array([func(x) for x in particles])
        g_best_index = np.argmin(p_best_values)
        g_best_position = p_best_positions[g_best_index]
        g_best_value = p_best_values[g_best_index]

        for i in range(self.budget // self.num_particles):
            w = self.w_max - (self.w_max - self.w_min) * i / (self.budget // self.num_particles)
            for j in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[j] = (w * velocities[j] +
                                 self.c1 * r1 * (p_best_positions[j] - particles[j]) +
                                 self.c2 * r2 * (g_best_position - particles[j]))
                particles[j] = np.clip(particles[j] + velocities[j], lb, ub)
                f = func(particles[j])

                if f < p_best_values[j]:
                    p_best_values[j] = f
                    p_best_positions[j] = particles[j]

                if f < g_best_value:
                    g_best_value = f
                    g_best_position = particles[j]
                    self.f_opt = g_best_value
                    self.x_opt = g_best_position

        return self.f_opt, self.x_opt