import numpy as np

class ImprovedAdaptivePSO:
    def __init__(self, budget=10000, dim=10, w_min=0.4, w_max=0.9, c1=2.0, c2=2.0, neighborhood_size=5):
        self.budget = budget
        self.dim = dim
        self.w_min = w_min
        self.w_max = w_max
        self.c1 = c1
        self.c2 = c2
        self.neighborhood_size = neighborhood_size
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
            inertia_weight = self.w_max - (_ / self.budget) * (self.w_max - self.w_min)
            velocities = inertia_weight * velocities + self.c1 * r1 * (pbest - swarm) + self.c2 * r2 * (gbest - swarm)
            swarm += velocities

            # Update personal best
            fit = np.array([func(x) for x in swarm])
            mask = fit < pbest_fit
            pbest_fit[mask] = fit[mask]
            pbest[mask] = swarm[mask]

            # Update global best within neighborhood
            for i in range(self.dim):
                neighborhood = np.random.choice(np.delete(np.arange(self.dim), i), self.neighborhood_size, replace=False)
                neighborhood_fitness = [func(swarm[j]) for j in neighborhood]
                best_neighbor = neighborhood[np.argmin(neighborhood_fitness)]
                if neighborhood_fitness[np.argmin(neighborhood_fitness)] < gbest_fit:
                    gbest_fit = neighborhood_fitness[np.argmin(neighborhood_fitness)]
                    gbest = swarm[best_neighbor]

        return gbest_fit, gbest