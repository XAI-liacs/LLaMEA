import numpy as np

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.iteration = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        num_particles = 30
        T_init = 1000
        T_min = 1
        cooling_rate = 0.97  # Change 1: Adjusted cooling rate

        particles = np.random.uniform(lb, ub, (num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (num_particles, self.dim))
        best_pos = particles.copy()
        best_val = np.array([func(p) for p in particles])
        g_best_val = np.min(best_val)
        g_best_pos = particles[np.argmin(best_val)]
        self.iteration += num_particles

        T = T_init
        while self.iteration < self.budget and T > T_min:
            for i in range(num_particles):
                if self.iteration >= self.budget:
                    break

                new_particle = particles[i] + np.random.uniform(-1, 1, self.dim) * T
                new_particle = np.clip(new_particle, lb, ub)
                new_val = func(new_particle)
                self.iteration += 1

                if new_val < best_val[i] or np.random.rand() < np.exp((best_val[i] - new_val) / T):
                    particles[i] = new_particle
                    best_val[i] = new_val
                    best_pos[i] = new_particle

                if new_val < g_best_val:
                    g_best_val = new_val
                    g_best_pos = new_particle

            for i in range(num_particles):
                r1, r2 = np.random.rand(2)
                inertia_weight = 0.9 - (0.5 * (self.iteration / self.budget))  # Change 2: Added adaptive inertia
                velocities[i] = (inertia_weight * velocities[i] +  # Change 3: Modified velocity update
                                 1.5 * r1 * (best_pos[i] - particles[i]) +
                                 1.5 * r2 * (g_best_pos - particles[i]))
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)

            T *= cooling_rate  # Change 4: Used adjusted cooling rate

        return g_best_pos, g_best_val