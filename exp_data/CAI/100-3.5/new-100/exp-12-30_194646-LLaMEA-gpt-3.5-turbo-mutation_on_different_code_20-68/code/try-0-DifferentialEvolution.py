import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            idxs = list(range(self.budget))
            idxs.remove(i)
            a, b, c = np.random.choice(idxs, 3, replace=False)
            mutant = np.clip(population[a] + self.mutation_factor * (population[b] - population[c]), func.bounds.lb, func.bounds.ub)
            crossover_points = np.random.rand(self.dim) < self.crossover_prob
            trial = np.where(crossover_points, mutant, population[i])
            
            f = func(trial)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = trial
                population[i] = trial
            
        return self.f_opt, self.x_opt