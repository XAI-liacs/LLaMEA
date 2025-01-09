import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def fitness(x):
            return func(x)

        pop_size = 30
        max_iter = self.budget // pop_size
        pop = np.random.uniform(-5.0, 5.0, size=(pop_size, self.dim))
        velocity = np.random.uniform(-1, 1, size=(pop_size, self.dim))
        pbest = pop.copy()
        gbest = pbest[np.argmin([fitness(ind) for ind in pbest])]

        for _ in range(max_iter):
            for i in range(pop_size):
                r1, r2 = np.random.uniform(0, 1, size=2)
                velocity[i] = 0.5 * velocity[i] + 2 * r1 * (pbest[i] - pop[i]) + 2 * r2 * (gbest - pop[i])
                pop[i] = np.clip(pop[i] + velocity[i], -5.0, 5.0)
                if fitness(pop[i]) < fitness(pbest[i]):
                    pbest[i] = pop[i]
                    if fitness(pop[i]) < fitness(gbest):
                        gbest = pop[i]

        return gbest