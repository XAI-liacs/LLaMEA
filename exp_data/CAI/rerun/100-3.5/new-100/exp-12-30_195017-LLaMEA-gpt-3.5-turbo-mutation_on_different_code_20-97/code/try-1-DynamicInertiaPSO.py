import numpy as np
from scipy.stats import uniform

class DynamicInertiaPSO:
    def __init__(self, budget=10000, dim=10, swarm_size=30, w_max=0.9, w_min=0.4, c1=1.5, c2=1.5):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.f_opt = np.Inf
        self.x_opt = None

    def latin_hypercube_sampling(self, bounds, n_samples):
        lhs = uniform(loc=bounds.lb, scale=bounds.ub - bounds.lb).rvs((n_samples, self.dim))
        np.random.shuffle(lhs)
        return lhs

    def __call__(self, func):
        bounds = func.bounds
        swarm = self.latin_hypercube_sampling(bounds, self.swarm_size)
        velocities = np.zeros((self.swarm_size, self.dim))
        pbest = swarm.copy()
        pbest_fitness = np.array([func(p) for p in pbest])
        gbest_idx = np.argmin(pbest_fitness)
        gbest = pbest[gbest_idx].copy()
        gbest_fitness = pbest_fitness[gbest_idx]

        for t in range(1, self.budget + 1):
            w = self.w_max - (self.w_max - self.w_min) * t / self.budget
            r1, r2 = np.random.uniform(0, 1, size=(2, self.swarm_size, self.dim))
            velocities = w * velocities + self.c1 * r1 * (pbest - swarm) + self.c2 * r2 * (gbest - swarm)
            swarm += velocities
            swarm = np.clip(swarm, bounds.lb, bounds.ub)

            fitness = np.array([func(p) for p in swarm])
            improved = fitness < pbest_fitness
            pbest[improved] = swarm[improved]
            pbest_fitness[improved] = fitness[improved]

            new_gbest_idx = np.argmin(pbest_fitness)
            if pbest_fitness[new_gbest_idx] < gbest_fitness:
                gbest = pbest[new_gbest_idx].copy()
                gbest_fitness = pbest_fitness[new_gbest_idx]

        self.f_opt = gbest_fitness
        self.x_opt = gbest
        return self.f_opt, self.x_opt