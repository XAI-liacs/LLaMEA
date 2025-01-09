import numpy as np

class DynamicPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def fitness(position):
            return func(position)

        lb = -5.0
        ub = 5.0
        pop_size = 30
        max_iter = self.budget // pop_size

        inertia_weight = 0.7
        c1 = 2.0
        c2 = 2.0

        pop = np.random.uniform(lb, ub, (pop_size, self.dim))
        velocity = np.zeros((pop_size, self.dim))
        pbest = pop.copy()
        pbest_fitness = np.array([fitness(ind) for ind in pop])
        gbest_idx = np.argmin(pbest_fitness)
        gbest = pbest[gbest_idx].copy()

        for _ in range(max_iter):
            for i in range(pop_size):
                r1 = np.random.uniform(0, 1, self.dim)
                r2 = np.random.uniform(0, 1, self.dim)

                velocity[i] = inertia_weight * velocity[i] + c1 * r1 * (pbest[i] - pop[i]) + c2 * r2 * (gbest - pop[i])
                pop[i] = np.clip(pop[i] + velocity[i], lb, ub)

                fitness_val = fitness(pop[i])
                if fitness_val < pbest_fitness[i]:
                    pbest[i] = pop[i].copy()
                    pbest_fitness[i] = fitness_val

                    if fitness_val < pbest_fitness[gbest_idx]:
                        gbest_idx = i
                        gbest = pop[i].copy()

        return gbest