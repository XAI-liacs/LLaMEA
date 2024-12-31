import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=50, f=0.5, cr=0.9, adapt_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f = f
        self.cr = cr
        self.adapt_rate = adapt_rate
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.population_size, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for _ in range(self.budget):
            for i in range(self.population_size):
                target = population[i]
                indices = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                a, b, c = population[indices]
                
                mutant = a + self.f * (b - c)
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, target)
                
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                
            if np.random.rand() < self.adapt_rate:
                self.f = max(0.1, min(0.9, self.f + np.random.uniform(-0.1, 0.1)))
                self.cr = max(0.1, min(0.9, self.cr + np.random.uniform(-0.1, 0.1))

        return self.f_opt, self.x_opt