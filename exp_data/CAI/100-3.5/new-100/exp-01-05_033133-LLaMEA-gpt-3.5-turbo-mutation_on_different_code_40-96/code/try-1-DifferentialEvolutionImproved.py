import numpy as np

class DifferentialEvolutionImproved:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.dim, self.dim))
        fitness = np.array([func(ind) for ind in population])
        f_history = []

        for i in range(self.budget):
            F = np.clip(np.random.normal(self.F, 0.1), 0, 2)
            CR = np.clip(np.random.normal(self.CR, 0.1), 0, 1)

            for j in range(self.dim):
                idxs = list(range(self.dim))
                idxs.remove(j)
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = population[a] + F * (population[b] - population[c])
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, population[j])
                
                f = func(trial)
                if f < fitness[j]:
                    population[j] = trial
                    fitness[j] = f
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = trial
            f_history.append(self.f_opt)

        return self.f_opt, self.x_opt