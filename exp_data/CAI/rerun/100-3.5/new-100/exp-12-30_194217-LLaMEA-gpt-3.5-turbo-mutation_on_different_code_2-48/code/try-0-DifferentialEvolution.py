import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=30):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None
    
    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for _ in range(self.budget):
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                trial_vector = population[a] + self.F * (population[b] - population[c])
                mask = np.random.rand(self.dim) < self.CR
                trial_vector = np.where(mask, trial_vector, population[i])
                
                f = func(trial_vector)
                if f < fitness[i]:
                    fitness[i] = f
                    population[i] = trial_vector
                
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial_vector
        
        return self.f_opt, self.x_opt