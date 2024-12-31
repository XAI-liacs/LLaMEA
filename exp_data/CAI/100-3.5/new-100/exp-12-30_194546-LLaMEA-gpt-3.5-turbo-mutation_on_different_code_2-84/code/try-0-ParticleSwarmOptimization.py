import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, num_particles=30, inertia=0.5, phi_p=0.5, phi_g=0.5):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.inertia = inertia
        self.phi_p = phi_p
        self.phi_g = phi_g
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        particles = np.random.uniform(lb, ub, size=(self.num_particles, self.dim))
        velocities = np.zeros((self.num_particles, self.dim))
        p_best_positions = particles.copy()
        p_best_values = np.array([func(p) for p in particles])
        g_best_idx = np.argmin(p_best_values)
        g_best_value = p_best_values[g_best_idx]
        g_best_position = particles[g_best_idx].copy()

        for _ in range(self.budget):
            for i in range(self.num_particles):
                r_p = np.random.rand(self.dim)
                r_g = np.random.rand(self.dim)
                velocities[i] = self.inertia*velocities[i] + self.phi_p*r_p*(p_best_positions[i] - particles[i]) + self.phi_g*r_g*(g_best_position - particles[i])
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)

                f = func(particles[i])
                if f < p_best_values[i]:
                    p_best_values[i] = f
                    p_best_positions[i] = particles[i].copy()
                    if f < g_best_value:
                        g_best_value = f
                        g_best_position = particles[i].copy()

        self.f_opt = g_best_value
        self.x_opt = g_best_position
        return self.f_opt, self.x_opt