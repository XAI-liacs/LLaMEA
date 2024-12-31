import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.particles = 30  # Initial number of particles
        self.w = 0.9        # Inertia weight
        self.w_min = 0.4    # Minimum inertia weight
        self.c1 = 2.0       # Cognitive component
        self.c2 = 2.0       # Social component
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        x = np.random.uniform(lb, ub, (self.particles, self.dim))
        v = np.random.uniform(-np.abs(ub - lb), np.abs(ub - lb), (self.particles, self.dim))
        p_best = x.copy()
        p_best_val = np.array([func(p) for p in p_best])
        
        g_best = p_best[np.argmin(p_best_val)]
        g_best_val = np.min(p_best_val)

        for t in range(self.budget // max(1, self.particles)):
            for i in range(self.particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                v[i] = self.w * v[i] + self.c1 * r1 * (p_best[i] - x[i]) + self.c2 * r2 * (g_best - x[i])
                x[i] = np.clip(x[i] + v[i], lb, ub)

                f_i = func(x[i])
                if f_i < p_best_val[i]:
                    p_best[i] = x[i]
                    p_best_val[i] = f_i

            current_g_best_index = np.argmin(p_best_val)
            current_g_best_val = p_best_val[current_g_best_index]
            
            if current_g_best_val < g_best_val:
                g_best = p_best[current_g_best_index]
                g_best_val = current_g_best_val

            self.w = self.w_min + (0.9 - self.w_min) * ((self.budget // max(1, self.particles) - t) / (self.budget // max(1, self.particles)))
            
            # Dynamically adjust the number of particles based on remaining budget
            self.particles = max(5, int(30 * (1 - t / (self.budget // 30))))

        self.f_opt, self.x_opt = g_best_val, g_best
        return self.f_opt, self.x_opt