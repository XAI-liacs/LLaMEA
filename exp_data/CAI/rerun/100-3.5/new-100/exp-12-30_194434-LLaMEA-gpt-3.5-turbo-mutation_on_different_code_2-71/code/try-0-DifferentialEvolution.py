import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])

        for i in range(self.budget):
            for j in range(len(population)):
                indices = [idx for idx in range(len(population)) if idx != j]
                a, b, c = np.random.choice(indices, 3, replace=False)

                mutant = population[a] + self.F * (population[b] - population[c])
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, population[j])

                trial_fitness = func(trial)
                if trial_fitness < fitness[j]:
                    population[j] = trial
                    fitness[j] = trial_fitness

            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.f_opt:
                self.f_opt = fitness[best_idx]
                self.x_opt = population[best_idx]

        return self.f_opt, self.x_opt