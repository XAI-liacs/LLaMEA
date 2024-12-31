import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        best_fitness = np.inf
        best_solution = None
        
        for i in range(self.budget):
            for j in range(len(self.population)):
                target = self.population[j]
                idxs = [idx for idx in range(len(self.population)) if idx != j]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = a + self.F * (b - c)
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, target)
                
                f = func(trial)
                if f < best_fitness:
                    best_fitness = f
                    best_solution = trial
                    
                if f < func(target):
                    self.population[j] = trial
        
        return best_fitness, best_solution