import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, num_particles=30, c1=2.0, c2=2.0, w=0.7):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.c1 = c1  # cognitive coefficient
        self.c2 = c2  # social coefficient
        self.w = w    # inertia weight
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initialize position and velocity of particles
        x = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        v = np.random.uniform(-abs(ub - lb), abs(ub - lb), (self.num_particles, self.dim))
        p_best = x.copy()
        p_best_values = np.array([func(p) for p in p_best])

        g_best = p_best[np.argmin(p_best_values)]
        g_best_value = np.min(p_best_values)

        evals = self.num_particles

        while evals < self.budget:
            for i in range(self.num_particles):
                # Calculate new velocity
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive = self.c1 * r1 * (p_best[i] - x[i])
                social = self.c2 * r2 * (g_best - x[i])
                v[i] = self.w * v[i] + cognitive + social

                # Update position
                x[i] = x[i] + v[i]
                # Apply bounds
                x[i] = np.clip(x[i], lb, ub)

                # Evaluate particle
                f = func(x[i])
                evals += 1

                # Update personal and global best
                if f < p_best_values[i]:
                    p_best[i] = x[i]
                    p_best_values[i] = f

                    if f < g_best_value:
                        g_best = x[i]
                        g_best_value = f

                if evals >= self.budget:
                    break

        self.f_opt, self.x_opt = g_best_value, g_best
        return self.f_opt, self.x_opt