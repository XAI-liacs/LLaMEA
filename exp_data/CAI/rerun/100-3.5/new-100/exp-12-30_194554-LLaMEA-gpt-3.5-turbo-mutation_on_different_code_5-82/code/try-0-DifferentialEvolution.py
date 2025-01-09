import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, f=0.8, cr=0.9, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.f = f
        self.cr = cr
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None
        
    def generate_population(self, func):
        return np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
    
    def __call__(self, func):
        population = self.generate_population(func)
        
        for _ in range(self.budget):
            for i in range(self.pop_size):
                indices = np.random.choice(self.pop_size, size=3, replace=False)
                a, b, c = population[indices]
                
                mutant = population[a] + self.f * (population[b] - population[c])
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, population[i])
                
                f_trial = func(trial)
                if f_trial < func(population[i]):
                    population[i] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        
        return self.f_opt, self.x_opt