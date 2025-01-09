import numpy as np

class HybridPSOAdaptiveLocalSearch:
    def __init__(self, budget=10000, dim=10, num_particles=20, max_iter=100):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.f_opt = np.Inf
        self.x_opt = None

    def local_search(self, x, func, radius=0.1):
        # Adaptive local search around current solution x with adaptive radius
        for _ in range(5):
            new_x = x + np.random.uniform(-radius, radius, size=self.dim)
            new_x = np.clip(new_x, func.bounds.lb, func.bounds.ub)
            if func(new_x) < func(x):
                x = new_x
        return x

    def __call__(self, func):
        swarm = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.num_particles, self.dim))
        velocities = np.random.uniform(-0.1, 0.1, size=(self.num_particles, self.dim))

        for _ in range(self.max_iter):
            for j in range(self.num_particles):
                particle = swarm[j]
                velocity = velocities[j]
                new_particle = particle + velocity
                new_particle = np.clip(new_particle, func.bounds.lb, func.bounds.ub)

                if np.random.rand() < 0.1:
                    new_particle = self.local_search(new_particle, func, radius=0.1)

                f = func(new_particle)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = new_particle

                swarm[j] = new_particle
                velocities[j] = velocities[j] + 0.1 * (new_particle - swarm[j]) + 0.1 * np.random.randn(self.dim)

            if _ % 10 == 0:
                for j in range(self.num_particles):
                    swarm[j] = self.local_search(swarm[j], func, radius=0.1)

        return self.f_opt, self.x_opt