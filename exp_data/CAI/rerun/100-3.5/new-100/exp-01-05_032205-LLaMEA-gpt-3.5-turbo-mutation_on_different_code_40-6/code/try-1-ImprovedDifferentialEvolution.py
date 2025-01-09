import numpy as np

class ImprovedDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size_factor=2):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size_factor = pop_size_factor
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.budget * self.pop_size_factor, self.dim))
        for i in range(self.budget):
            trial_population = np.zeros_like(population)
            for j in range(self.budget * self.pop_size_factor):
                idxs = [idx for idx in range(self.budget * self.pop_size_factor) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
                jrand = np.random.randint(self.dim)
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover | (np.arange(self.dim) == jrand), mutant, population[i])
                if func(trial) < func(population[i]):
                    trial_population[j] = trial
                else:
                    trial_population[j] = population[i]
            population = trial_population
        self.x_opt = population[np.argmin([func(x) for x in population])]
        self.f_opt = func(self.x_opt)

        return self.f_opt, self.x_opt