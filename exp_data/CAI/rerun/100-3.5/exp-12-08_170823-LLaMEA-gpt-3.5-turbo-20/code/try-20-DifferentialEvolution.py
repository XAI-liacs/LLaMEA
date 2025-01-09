import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.CR = 0.9  # Crossover rate
        self.F = 0.5   # Differential weight
        self.CR_adaptive = np.full(self.budget, self.CR)  # Self-adaptive CR
        self.F_adaptive = np.full(self.budget, self.F)    # Self-adaptive F
        
    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for i in range(self.budget):
            for j in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                self.CR_adaptive[j] = np.clip(np.random.normal(self.CR, 0.1), 0, 1)  # Update CR adaptively
                self.F_adaptive[j] = np.abs(np.random.normal(self.F, 0.1))           # Update F adaptively
                
                mutant = np.clip(a + self.F_adaptive[j] * (b - c), func.bounds.lb, func.bounds.ub)
                
                crossover_mask = np.random.rand(self.dim) < self.CR_adaptive[j]
                trial = np.where(crossover_mask, mutant, population[j])
                
                f_trial = func(trial)
                if f_trial < fitness[j]:
                    population[j] = trial
                    fitness[j] = f_trial
                    
                if fitness[j] < self.f_opt:
                    self.f_opt = fitness[j]
                    self.x_opt = population[j]

        return self.f_opt, self.x_opt