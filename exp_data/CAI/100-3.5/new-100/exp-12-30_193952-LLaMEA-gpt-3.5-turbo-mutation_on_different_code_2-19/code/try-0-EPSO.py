import numpy as np

class EPSO:
    def __init__(self, budget=10000, dim=10, swarm_size=20, w=0.5, c1=2.0, c2=2.0, p=0.5):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.p = p
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        swarm = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        p_best = swarm.copy()
        p_best_f = np.array([func(x) for x in swarm])
        g_best_idx = np.argmin(p_best_f)
        g_best = swarm[g_best_idx].copy()

        for i in range(self.budget):
            r1, r2 = np.random.rand(2, self.swarm_size, self.dim)
            velocities = self.w * velocities + self.c1 * r1 * (p_best - swarm) + self.c2 * r2 * (g_best - swarm)
            velocities = np.clip(velocities, func.bounds.lb, func.bounds.ub)
            swarm += velocities

            for j, x in enumerate(swarm):
                f = func(x)
                if f < p_best_f[j]:
                    p_best[j] = x
                    p_best_f[j] = f
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = x

            if np.random.rand() < self.p:  # Neighborhood search
                neighbor_idx = np.random.choice(np.delete(np.arange(self.swarm_size), g_best_idx))
                swarm[g_best_idx] = swarm[neighbor_idx]
                g_best = swarm[g_best_idx]

        return self.f_opt, self.x_opt