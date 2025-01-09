import numpy as np

class EvolutionaryAlgorithm_DE_Cauchy:
    def __init__(self, budget=10000, dim=10, population_size=50, f=0.5, cr=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f = f
        self.cr = cr
        self.f_opt = np.Inf
        self.x_opt = None

    def cauchy_mutation(self, x, x_best, scale=0.1):
        return x_best + scale * np.tan(np.pi * (x - 0.5))

    def differential_evolution(self, population, func):
        for i in range(self.population_size):
            x = population[i]
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = np.random.choice(indices, 3, replace=False)
            
            mutant = self.cauchy_mutation(population[a], population[b])
            trial = np.where(np.random.uniform(0, 1, self.dim) < self.cr, mutant, x)
            
            f_trial = func(trial)
            if f_trial < func(x):
                population[i] = trial
        
        return population

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        
        for _ in range(self.budget // self.population_size):
            population = self.differential_evolution(population, func)
            
            for x in population:
                f = func(x)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = x

        return self.f_opt, self.x_opt