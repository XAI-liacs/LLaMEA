import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, swarm_size=50):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.c1 = 2.5  # personal acceleration coefficient
        self.c2 = 0.5  # global acceleration coefficient
        self.w_max = 0.9  # maximum inertia weight
        self.w_min = 0.4  # minimum inertia weight

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        V_max = (ub - lb) * 0.1
        X = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        V = np.random.uniform(-V_max, V_max, (self.swarm_size, self.dim))
        p_best_X = np.copy(X)
        p_best_F = np.array([func(x) for x in X])

        g_best_index = np.argmin(p_best_F)
        g_best_X = p_best_X[g_best_index]
        g_best_F = p_best_F[g_best_index]

        evaluations = self.swarm_size

        while evaluations < self.budget:
            w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)
            r1, r2 = np.random.rand(2, self.swarm_size, 1)
            
            V = w * V + self.c1 * r1 * (p_best_X - X) + self.c2 * r2 * (g_best_X - X)
            V = np.clip(V, -V_max, V_max)
            X = np.clip(X + V, lb, ub)

            F = np.array([func(x) for x in X])
            evaluations += self.swarm_size

            better_indices = F < p_best_F
            p_best_X[better_indices] = X[better_indices]
            p_best_F[better_indices] = F[better_indices]

            g_best_index = np.argmin(p_best_F)
            g_best_X = p_best_X[g_best_index]
            g_best_F = p_best_F[g_best_index]

            if g_best_F < self.f_opt:
                self.f_opt = g_best_F
                self.x_opt = g_best_X

        return self.f_opt, self.x_opt