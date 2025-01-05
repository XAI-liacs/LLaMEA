import numpy as np

class EnhancedDE:
    def __init__(self, budget=10000, dim=10, cr=0.5, f=0.5, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.cr = cr
        self.f = f
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in pop])

        for i in range(self.budget):
            mutant_pop = pop + self.f * (pop[np.random.choice(range(self.pop_size), size=(self.pop_size, 3))] - pop)
            crossover = np.random.rand(self.pop_size, self.dim) < self.cr
            trial_pop = np.where(crossover, mutant_pop, pop)
            trial_fitness = np.array([func(x) for x in trial_pop])

            for j in range(self.pop_size):
                if trial_fitness[j] < fitness[j]:
                    pop[j] = trial_pop[j]
                    fitness[j] = trial_fitness[j]

            sorted_indices = np.argsort(fitness)
            worst_indices = sorted_indices[int(0.9 * self.pop_size):]
            for idx in worst_indices:
                pop[idx] = np.random.uniform(func.bounds.lb, func.bounds.ub)
                fitness[idx] = func(pop[idx])

            if np.min(fitness) < self.f_opt:
                idx = np.argmin(fitness)
                self.f_opt = fitness[idx]
                self.x_opt = pop[idx]

        return self.f_opt, self.x_opt