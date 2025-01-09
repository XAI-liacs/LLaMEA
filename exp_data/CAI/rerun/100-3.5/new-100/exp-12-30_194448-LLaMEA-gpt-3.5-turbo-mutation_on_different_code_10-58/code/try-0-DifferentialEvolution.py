import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        
        for i in range(self.budget):
            for j in range(self.pop_size):
                target = population[j]
                
                idxs = [idx for idx in range(self.pop_size) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                mutant = a + self.F * (b - c)
                for k in range(self.dim):
                    if np.random.rand() > self.CR:
                        mutant[k] = target[k]
                
                mutant_fitness = func(mutant)
                if mutant_fitness < func(target):
                    population[j] = mutant
                    
                    if mutant_fitness < self.f_opt:
                        self.f_opt = mutant_fitness
                        self.x_opt = mutant
            
        return self.f_opt, self.x_opt