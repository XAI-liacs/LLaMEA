import numpy as np

class DynamicDE:
    def __init__(self, budget=10000, dim=10, CR=0.5, F=0.5):
        self.budget = budget
        self.dim = dim
        self.CR = CR
        self.F = F
        self.f_opt = np.Inf
        self.x_opt = None

    def mutate(self, population, target_idx):
        candidates = [idx for idx in range(len(population)) if idx != target_idx]
        a, b, c = np.random.choice(candidates, 3, replace=False)
        mutant = population[a] + self.F * (population[b] - population[c])
        return mutant

    def crossover(self, target, mutant):
        crossover_points = np.random.rand(self.dim) < self.CR
        trial = np.where(crossover_points, mutant, target)
        return trial

    def adapt_parameters(self, iteration):
        self.F = 0.5 + 0.3 * np.sin(0.2 * np.pi * iteration)
        self.CR = 0.5 + 0.1 * np.cos(0.4 * np.pi * iteration)

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for itr in range(self.budget):
            self.adapt_parameters(itr)
            for target_idx in range(len(population)):
                mutant = self.mutate(population, target_idx)
                trial = self.crossover(population[target_idx], mutant)

                f = func(trial)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial

        return self.f_opt, self.x_opt