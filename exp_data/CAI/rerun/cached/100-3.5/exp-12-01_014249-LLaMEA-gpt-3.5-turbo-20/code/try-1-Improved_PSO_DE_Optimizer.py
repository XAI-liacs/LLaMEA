import numpy as np

class Improved_PSO_DE_Optimizer:
    def __init__(self, budget, dim, swarm_size=30, de_mut=0.5, de_crossp=0.7, c1=1.5, c2=1.5, w_max=0.9, w_min=0.4):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.de_mut = de_mut
        self.de_crossp = de_crossp
        self.c1 = c1
        self.c2 = c2
        self.w_max = w_max
        self.w_min = w_min

    def __call__(self, func):
        def constrain(x):
            return np.clip(x, -5.0, 5.0)

        swarm = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        p_best = swarm.copy()
        p_best_fitness = np.array([func(ind) for ind in p_best])
        g_best_idx = np.argmin(p_best_fitness)
        g_best = p_best[g_best_idx].copy()

        for iter_count in range(1, self.budget + 1):
            w = self.w_max - (self.w_max - self.w_min) * iter_count / self.budget
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = w * velocities[i] + self.c1 * r1 * (p_best[i] - swarm[i]) + self.c2 * r2 * (g_best - swarm[i])
                swarm[i] += velocities[i]
                swarm[i] = constrain(swarm[i])
                if np.random.rand() < self.de_crossp:
                    idxs = np.random.choice(self.swarm_size, 3, replace=False)
                    mutant = swarm[idxs[0]] + self.de_mut * (swarm[idxs[1]] - swarm[idxs[2])
                    crossover_points = np.random.rand(self.dim) < self.de_crossp
                    swarm[i] = np.where(crossover_points, mutant, swarm[i])
                swarm[i] = constrain(swarm[i])
                fitness = func(swarm[i])
                if fitness < p_best_fitness[i]:
                    p_best[i] = swarm[i].copy()
                    p_best_fitness[i] = fitness
                    if fitness < p_best_fitness[g_best_idx]:
                        g_best_idx = i
                        g_best = swarm[i].copy()

        return g_best