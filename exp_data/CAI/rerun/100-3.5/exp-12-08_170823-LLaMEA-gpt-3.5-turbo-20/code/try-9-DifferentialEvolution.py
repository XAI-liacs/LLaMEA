import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.CR = 0.9  # Crossover rate
        self.F = 0.5   # Differential weight
        
    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        CRs = np.random.uniform(0.1, 1.0, self.budget)  # Initialize individual CRs
        Fs = np.random.uniform(0.1, 0.9, self.budget)   # Initialize individual F values
        
        for i in range(self.budget):
            for j in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                CR = CRs[j]  # Individual crossover rate
                F = Fs[j]    # Individual differential weight
                
                mutant = np.clip(a + F * (b - c), func.bounds.lb, func.bounds.ub)
                
                crossover_mask = np.random.rand(self.dim) < CR
                trial = np.where(crossover_mask, mutant, population[j])
                
                f_trial = func(trial)
                if f_trial < fitness[j]:
                    population[j] = trial
                    fitness[j] = f_trial
                    
                    if np.random.rand() < 0.1:  # Randomly update CR and F
                        CRs[j] = np.clip(CR + 0.1 * np.random.randn(), 0.1, 1.0)
                        Fs[j] = np.clip(F + 0.1 * np.random.randn(), 0.1, 0.9)
                    
                if fitness[j] < self.f_opt:
                    self.f_opt = fitness[j]
                    self.x_opt = population[j]

        return self.f_opt, self.x_opt