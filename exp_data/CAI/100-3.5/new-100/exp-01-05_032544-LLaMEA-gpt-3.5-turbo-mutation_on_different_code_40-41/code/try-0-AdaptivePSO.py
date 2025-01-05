import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, w=0.5, c1=2.0, c2=2.0):
        self.budget = budget
        self.dim = dim
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.f_opt = np.Inf
        self.x_opt = np.random.uniform(-5.0, 5.0, dim)

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, (self.dim, self.dim))
        velocities = np.zeros((self.dim, self.dim))
        pbest = swarm
        pbest_fit = np.array([func(x) for x in swarm])
        gbest = pbest[np.argmin(pbest_fit)]
        gbest_fit = np.min(pbest_fit)

        for _ in range(self.budget):
            r1, r2 = np.random.rand(self.dim, 2)
            velocities = self.w * velocities + self.c1 * r1 * (pbest - swarm) + self.c2 * r2 * (gbest - swarm)
            swarm += velocities

            # Update personal best
            fit = np.array([func(x) for x in swarm])
            mask = fit < pbest_fit
            pbest_fit[mask] = fit[mask]
            pbest[mask] = swarm[mask]

            # Update global best
            gbest_idx = np.argmin(pbest_fit)
            if pbest_fit[gbest_idx] < gbest_fit:
                gbest_fit = pbest_fit[gbest_idx]
                gbest = pbest[gbest_idx]

        return gbest_fit, gbest