import numpy as np

class DE_PSO_Optimizer:
    def __init__(self, budget, dim, pop_size=50, max_iter=1000, F=0.5, CR=0.9, w=0.5, c1=2.0, c2=2.0):
        self.budget, self.dim, self.pop_size, self.max_iter, self.F, self.CR, self.w, self.c1, self.c2 = budget, dim, pop_size, max_iter, F, CR, w, c1, c2

    def __call__(self, func):
        def de_pso(func):
            population, pbest_pos, velocities = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim)), None, np.zeros((self.pop_size, self.dim))
            gbest_pos = population[0].copy()
            gbest_val = func(gbest_pos)

            for _ in range(self.max_iter):
                pbest_pos = population if pbest_pos is None else pbest_pos
                pbest_val = np.array([func(ind) for ind in population])

                for i in range(self.pop_size):
                    r1, r2 = np.random.rand(), np.random.rand()
                    velocities[i] = self.w * velocities[i] + self.c1 * r1 * (pbest_pos[i] - population[i]) + self.c2 * r2 * (gbest_pos - population[i])
                    population[i] = np.clip(population[i] + velocities[i], -5.0, 5.0)
                    new_val = func(population[i])

                    if new_val < pbest_val[i]:
                        pbest_val[i], pbest_pos[i] = new_val, population[i].copy()
                        if new_val < gbest_val:
                            gbest_pos, gbest_val = pbest_pos[i].copy(), new_val

                if func.calls >= self.budget:
                    break

            return gbest_pos

        return de_pso(func)