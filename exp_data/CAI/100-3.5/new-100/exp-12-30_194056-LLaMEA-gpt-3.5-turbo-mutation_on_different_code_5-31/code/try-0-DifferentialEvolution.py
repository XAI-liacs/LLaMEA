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
            target_idx = np.random.randint(self.budget)
            indices = np.random.choice(np.delete(np.arange(self.budget), target_idx, axis=0), size=2, replace=False)
            a, b, c = population[indices]
            mutant = population[target_idx] + self.F * (a - b)
            crossover = np.random.rand(self.dim) < self.CR
            trial = np.where(crossover, mutant, population[target_idx])
            
            f_target = func(population[target_idx])
            f_trial = func(trial)
            if f_trial < f_target:
                population[target_idx] = trial
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

        return self.f_opt, self.x_opt