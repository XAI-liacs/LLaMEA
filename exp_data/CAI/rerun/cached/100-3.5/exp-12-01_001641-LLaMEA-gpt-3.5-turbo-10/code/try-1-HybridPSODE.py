import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        pop_size = 10
        p = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        v = np.zeros((pop_size, self.dim))
        p_best = p.copy()
        p_best_fit = np.array([func(x) for x in p])
        g_best_idx = np.argmin(p_best_fit)
        g_best = p[g_best_idx].copy()
        w = 0.5
        c1 = 1.5
        c2 = 1.5

        for _ in range(self.budget):
            r1 = np.random.random((pop_size, self.dim))
            r2 = np.random.random((pop_size, self.dim))
            v = w * v + c1 * r1 * (p_best - p) + c2 * r2 * (g_best - p)
            p = p + v
            p = np.clip(p, -5.0, 5.0)
            fitness = np.array([func(x) for x in p])
            better_idx = fitness < p_best_fit
            p_best[better_idx] = p[better_idx]
            p_best_fit[better_idx] = fitness[better_idx]
            g_best_idx = np.argmin(p_best_fit)
            g_best = p[g_best_idx]

        return g_best