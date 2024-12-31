import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            for j in range(self.budget):
                a, b, c = np.random.choice(population, 3, replace=False)
                F_adaptive = np.clip(np.random.normal(self.F, 0.1), 0, 2)  # Adaptive control of F
                CR_adaptive = np.clip(np.random.normal(self.CR, 0.1), 0, 1)  # Adaptive control of CR
                mutant = a + F_adaptive * (b - c)
                crossover = np.random.rand(self.dim) < CR_adaptive
                trial = np.where(crossover, mutant, population[i])
                
                f = func(trial)
                if f < func(population[i]):
                    population[i] = trial
                    
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial
            
        return self.f_opt, self.x_opt