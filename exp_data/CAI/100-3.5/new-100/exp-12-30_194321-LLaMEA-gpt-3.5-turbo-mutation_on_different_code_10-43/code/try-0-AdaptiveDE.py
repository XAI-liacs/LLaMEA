import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.CR = 0.5
        self.F = 0.5

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])

        for i in range(self.budget):
            for j in range(len(population)):
                idxs = [idx for idx in range(len(population)) if idx != j]
                a, b, c = np.random.choice(idxs, 3, replace=False)

                mutant = population[a] + self.F * (population[b] - population[c])
                crossover_points = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_points, mutant, population[j])
                
                f = func(trial)
                if f < fitness[j]:
                    population[j] = trial
                    fitness[j] = f

            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.f_opt:
                self.f_opt = fitness[best_idx]
                self.x_opt = population[best_idx]
                
            if i % 10 == 0:  # Adaptive parameter adjustment every 10 iterations
                self.CR = max(0.1, min(0.9, self.CR + 0.01 * (fitness[best_idx] - self.f_opt)))
                self.F = max(0.1, min(0.9, self.F + 0.01 * (fitness[best_idx] - self.f_opt)))

        return self.f_opt, self.x_opt