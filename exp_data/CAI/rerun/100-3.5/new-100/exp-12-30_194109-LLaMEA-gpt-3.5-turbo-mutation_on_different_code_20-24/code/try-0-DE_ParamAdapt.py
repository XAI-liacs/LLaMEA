import numpy as np

class DE_ParamAdapt:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for i in range(self.budget):
            for j in range(len(population)):
                indices = np.random.choice(len(population), 3, replace=False)
                x_r1, x_r2, x_r3 = population[indices]
                
                mutant_vector = population[j] + self.F * (x_r1 - population[j]) + self.F * (x_r2 - x_r3)
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial_vector = np.where(crossover_mask, mutant_vector, population[j])
                
                f_trial = func(trial_vector)
                if f_trial < fitness[j]:
                    population[j] = trial_vector
                    fitness[j] = f_trial
                
                if fitness[j] < self.f_opt:
                    self.f_opt = fitness[j]
                    self.x_opt = population[j]
        
        return self.f_opt, self.x_opt