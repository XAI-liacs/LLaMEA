import numpy as np

class EnhancedAdaptivePSO:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.w_min = 0.4  # Minimum inertia weight
        self.w_max = 0.9  # Maximum inertia weight
        self.c_min = 1.5  # Minimum parameter
        self.c_max = 2.5  # Maximum parameter

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
        
        w, c1, c2 = self.w_max, self.c_max, self.c_max

        for _ in range(self.budget):
            r1 = np.random.uniform(0, 1, (swarm_size, self.dim))
            r2 = np.random.uniform(0, 1, (swarm_size, self.dim))

            velocities = w * velocities + c1 * r1 * (p_best - swarm) + c2 * r2 * (g_best - swarm)
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

            # Dynamic parameter adaptation based on performance
            w = self.w_min + (_ / self.budget) * (self.w_max - self.w_min)
            c1 = self.c_max - (_ / self.budget) * (self.c_max - self.c_min)
            c2 = self.c_min + (_ / self.budget) * (self.c_max - self.c_min)
            
        self.f_opt = g_best_f
        self.x_opt = g_best
        return self.f_opt, self.x_opt