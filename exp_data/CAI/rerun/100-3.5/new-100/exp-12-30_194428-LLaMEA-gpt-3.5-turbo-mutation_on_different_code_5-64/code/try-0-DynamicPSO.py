import numpy as np

class DynamicPSO:
    def __init__(self, budget=10000, dim=10, w_max=0.9, w_min=0.4, c1=2.0, c2=2.0):
        self.budget = budget
        self.dim = dim
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, (self.dim, self.dim))
        velocities = np.zeros((self.dim, self.dim))
        pbest = swarm.copy()
        pbest_f = np.array([func(p) for p in pbest])
        gbest_idx = np.argmin(pbest_f)
        gbest = pbest[gbest_idx].copy()
        gbest_f = pbest_f[gbest_idx]

        inertia_weight = self.w_max
        for i in range(self.budget):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities = inertia_weight * velocities + self.c1 * r1 * (pbest - swarm) + self.c2 * r2 * (gbest - swarm)
            swarm += velocities
            swarm = np.clip(swarm, -5.0, 5.0)
            
            current_f = np.array([func(p) for p in swarm])
            update_indices = current_f < pbest_f
            pbest[update_indices] = swarm[update_indices]
            pbest_f[update_indices] = current_f[update_indices]
            
            gbest_idx = np.argmin(pbest_f)
            if pbest_f[gbest_idx] < gbest_f:
                gbest = pbest[gbest_idx].copy()
                gbest_f = pbest_f[gbest_idx]
                
            inertia_weight = self.w_max - (self.w_max - self.w_min) * i / self.budget

        self.f_opt = gbest_f
        self.x_opt = gbest
        return self.f_opt, self.x_opt