import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, num_particles=30):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.f_opt = np.inf
        self.x_opt = None
        self.c1 = 2.0  # cognitive coefficient
        self.c2 = 2.0  # social coefficient
        self.w_max = 0.9
        self.w_min = 0.4

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        x = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        v = np.zeros((self.num_particles, self.dim))
        p_best = x.copy()
        p_best_val = np.array([func(xi) for xi in x])
        g_best = p_best[np.argmin(p_best_val)]
        g_best_val = np.min(p_best_val)

        evals = self.num_particles

        def levy_flight(Lambda):
            sigma_u = np.power((np.math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2)) /
                               (np.math.gamma((1 + Lambda) / 2) * Lambda * np.power(2, (Lambda - 1) / 2)), 1 / Lambda)
            u = np.random.normal(0, sigma_u, self.dim)
            v = np.random.normal(0, 1, self.dim)
            step = u / np.power(np.abs(v), 1 / Lambda)
            return 0.01 * step

        while evals < self.budget:
            w = self.w_max - ((self.w_max - self.w_min) * evals / self.budget)

            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                v[i] = w * v[i] + self.c1 * r1 * (p_best[i] - x[i]) + self.c2 * r2 * (g_best - x[i])
                v[i] = np.clip(v[i], -abs(ub - lb), abs(ub - lb))
                if np.random.rand() < 0.1:
                    x[i] += levy_flight(1.5)  # Lévy flight step
                else:
                    x[i] = x[i] + v[i]
                x[i] = np.clip(x[i], lb, ub)

                f_value = func(x[i])
                evals += 1

                if f_value < p_best_val[i]:
                    p_best[i] = x[i].copy()
                    p_best_val[i] = f_value

                if f_value < g_best_val:
                    g_best = x[i].copy()
                    g_best_val = f_value
                    if f_value < self.f_opt:
                        self.f_opt = f_value
                        self.x_opt = x[i].copy()

                if evals >= self.budget:
                    break

        return self.f_opt, self.x_opt