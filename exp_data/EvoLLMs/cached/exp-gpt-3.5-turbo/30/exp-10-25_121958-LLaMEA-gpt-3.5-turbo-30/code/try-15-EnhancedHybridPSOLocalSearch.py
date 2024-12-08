import numpy as np

class EnhancedHybridPSOLocalSearch:
    def __init__(self, budget, dim, n_particles=30, max_local_iter=10):
        self.budget = budget
        self.dim = dim
        self.n_particles = n_particles
        self.max_local_iter = max_local_iter

    def __call__(self, func):
        def local_search(x):
            best_x = x.copy()
            best_fitness = func(x)
            for _ in range(self.max_local_iter):
                perturbation = np.random.uniform(-0.1, 0.1, size=self.dim)
                new_x = x + perturbation
                new_fitness = func(new_x)
                if new_fitness < best_fitness:
                    best_x = new_x
                    best_fitness = new_fitness
            return best_x

        swarm = np.random.uniform(-5.0, 5.0, size=(self.n_particles, self.dim))
        swarm_fitness = np.array([func(p) for p in swarm])
        best_idx = np.argmin(swarm_fitness)
        best_solution = swarm[best_idx]

        for _ in range(self.budget // self.n_particles):
            for i in range(self.n_particles):
                exploration_factor = np.random.uniform(-1, 1, size=self.dim)
                new_position = swarm[i] + exploration_factor * (best_solution - swarm[i])
                new_position = np.clip(new_position, -5.0, 5.0)
                new_position = local_search(new_position)
                new_fitness = func(new_position)

                if new_fitness < swarm_fitness[i]:
                    swarm[i] = new_position
                    swarm_fitness[i] = new_fitness

                    if new_fitness < func(best_solution):
                        best_solution = new_position

        return best_solution