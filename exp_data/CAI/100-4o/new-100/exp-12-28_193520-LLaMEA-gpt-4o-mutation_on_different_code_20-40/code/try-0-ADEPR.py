import numpy as np

class ADEPR:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None
        self.F = 0.5
        self.CR = 0.9
        self.pop_size = 10 * dim
        self.pop = None

    def initialize_population(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))

    def mutate(self, target_idx, bounds):
        indices = list(range(self.pop_size))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.pop[a] + self.F * (self.pop[b] - self.pop[c])
        return np.clip(mutant, bounds.lb, bounds.ub)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        crossover_mask[np.random.randint(self.dim)] = True
        return np.where(crossover_mask, mutant, target)

    def select(self, candidate, target_idx, func):
        f_candidate = func(candidate)
        if f_candidate < self.f_opt:
            self.f_opt = f_candidate
            self.x_opt = candidate
        if f_candidate < self.pop_fitness[target_idx]:
            self.pop[target_idx] = candidate
            self.pop_fitness[target_idx] = f_candidate

    def adapt_parameters(self):
        self.F = np.random.uniform(0.4, 0.9)
        self.CR = np.random.uniform(0.1, 1.0)
        
    def resize_population(self, iteration):
        if iteration % 100 == 0:
            self.pop_size = max(4, int(self.pop_size * 0.9))
            if self.pop_size < len(self.pop):
                self.pop = self.pop[:self.pop_size]
                self.pop_fitness = self.pop_fitness[:self.pop_size]
            else:
                extra_pop = self.initialize_population(self.func_bounds)
                extra_fitness = np.array([self.func(x) for x in extra_pop])
                self.pop = np.vstack((self.pop, extra_pop))
                self.pop_fitness = np.hstack((self.pop_fitness, extra_fitness))

    def __call__(self, func):
        self.func_bounds = func.bounds
        self.pop = self.initialize_population(func.bounds)
        self.pop_fitness = np.array([func(ind) for ind in self.pop])

        for iteration in range(self.budget):
            for i in range(len(self.pop)):
                mutant = self.mutate(i, func.bounds)
                trial = self.crossover(self.pop[i], mutant)
                self.select(trial, i, func)

            self.adapt_parameters()
            self.resize_population(iteration)

        return self.f_opt, self.x_opt