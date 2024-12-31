import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def evolve_population(self, pop, func):
        new_pop = np.copy(pop)
        for i in range(len(pop)):
            candidates = [idx for idx in range(len(pop)) if idx != i]
            a, b, c = pop[np.random.choice(candidates, 3, replace=False)]
            mutant = pop[i] + self.F * (a - pop[i]) + self.F * (b - c)
            for j in range(self.dim):
                if np.random.rand() > self.CR:
                    mutant[j] = pop[i][j]
            f_mutant = func(mutant)
            if f_mutant < func(pop[i]):
                new_pop[i] = mutant
        return new_pop

    def __call__(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for _ in range(self.budget):
            pop = self.evolve_population(pop, func)
            best_idx = np.argmin([func(ind) for ind in pop])
            if func(pop[best_idx]) < self.f_opt:
                self.f_opt = func(pop[best_idx])
                self.x_opt = pop[best_idx]
        return self.f_opt, self.x_opt