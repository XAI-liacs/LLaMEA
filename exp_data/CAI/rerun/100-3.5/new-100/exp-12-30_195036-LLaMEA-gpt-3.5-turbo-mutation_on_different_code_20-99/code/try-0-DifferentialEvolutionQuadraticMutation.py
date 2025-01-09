import numpy as np

class DifferentialEvolutionQuadraticMutation:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None
    
    def quadratic_mutation(self, population, F):
        v = np.zeros_like(population)
        for i in range(len(population)):
            idxs = np.random.choice(np.setdiff1d(np.arange(len(population)), i, assume_unique=True), 2, replace=False)
            v[i] = population[i] + F * (population[idxs[0]] - population[idxs[1]])
        return v
    
    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.dim, self.dim))
        
        for i in range(self.budget):
            v = self.quadratic_mutation(population, self.F)
            crossover = np.random.rand(self.dim, self.dim) < self.CR
            trial_population = np.where(crossover, v, population)
            
            f_vals = np.array([func(x) for x in trial_population])
            min_idx = np.argmin(f_vals)
            if f_vals[min_idx] < self.f_opt:
                self.f_opt = f_vals[min_idx]
                self.x_opt = trial_population[min_idx]
            
            population = trial_population
        
        return self.f_opt, self.x_opt