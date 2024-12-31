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
        for i in range(self.budget):
            for j in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != i and idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                F_i = np.random.uniform(0, 1) if np.random.rand() > 0.2 else self.F
                CR_i = np.random.uniform(0, 1) if np.random.rand() > 0.2 else self.CR
                mutant = population[a] + F_i * (population[b] - population[c])
                cross_points = np.random.rand(self.dim) < CR_i
                offspring = np.where(cross_points, mutant, population[i])
                f_offspring = func(offspring)
                if f_offspring < func(population[i]):
                    population[i] = offspring
                    if f_offspring < self.f_opt:
                        self.f_opt = f_offspring
                        self.x_opt = offspring
        return self.f_opt, self.x_opt