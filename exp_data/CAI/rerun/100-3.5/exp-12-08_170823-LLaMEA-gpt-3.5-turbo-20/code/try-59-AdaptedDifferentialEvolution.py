import numpy as np

class AdaptedDifferentialEvolution:
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
        
        for i in range(self.budget):
            for j in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
                
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, population[j])
                
                f_trial = func(trial)
                if f_trial < fitness[j]:
                    population[j] = trial
                    fitness[j] = f_trial
                    
                    if f_trial < self.f_opt:  # Update if the trial is better than the global optimum
                        self.f_opt = f_trial
                        self.x_opt = trial
                        
            if i % 100 == 0:  # Adaptively adjust control parameters every 100 iterations
                self.F = max(0.1, self.F * 0.95)  # Decrease differential weight
                self.CR = min(0.95, self.CR * 1.05)  # Increase crossover rate

        return self.f_opt, self.x_opt