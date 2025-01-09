import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.pop_size = 10
        self.F = 0.5
        self.CR = 0.9

    def mutation(self, population, F):
        idxs = np.random.choice(len(population), 3, replace=False)
        a, b, c = population[idxs]
        mutant = a + F * (b - c)
        return mutant

    def crossover(self, target, mutant, CR):
        trial = np.copy(target)
        for i in range(len(target)):
            if np.random.rand() > CR:
                trial[i] = mutant[i]
        return trial

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        
        for _ in range(self.budget):
            for i in range(self.pop_size):
                target = population[i]
                mutant = self.mutation(population, self.F)
                trial = self.crossover(target, mutant, self.CR)
                
                f_target = func(target)
                f_trial = func(trial)
                
                if f_trial < f_target:
                    population[i] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
        return self.f_opt, self.x_opt