import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.w = 0.5  # inertia weight
        self.c1 = 2.0  # cognitive parameter
        self.c2 = 2.0  # social parameter

    def __call__(self, func):
        swarm_size = 20
        lb = func.bounds.lb
        ub = func.bounds.ub
        swarm = np.random.uniform(lb, ub, size=(swarm_size, self.dim))
        velocities = np.zeros((swarm_size, self.dim))
        p_best = swarm.copy()
        p_best_f = np.array([func(x) for x in swarm])
        g_best_idx = np.argmin(p_best_f)
        g_best = swarm[g_best_idx].copy()
        g_best_f = p_best_f[g_best_idx]

        for _ in range(self.budget):
            r1 = np.random.uniform(0, 1, (swarm_size, self.dim))
            r2 = np.random.uniform(0, 1, (swarm_size, self.dim))

            velocities = self.w * velocities + self.c1 * r1 * (p_best - swarm) + self.c2 * r2 * (g_best - swarm)
            swarm = swarm + velocities
            swarm = np.clip(swarm, lb, ub)

            f_vals = np.array([func(x) for x in swarm])
            updates = f_vals < p_best_f
            p_best[updates] = swarm[updates]
            p_best_f[updates] = f_vals[updates]

            g_best_idx = np.argmin(p_best_f)
            if p_best_f[g_best_idx] < g_best_f:
                g_best = p_best[g_best_idx].copy()
                g_best_f = p_best_f[g_best_idx]

        self.f_opt = g_best_f
        self.x_opt = g_best
        return self.f_opt, self.x_opt