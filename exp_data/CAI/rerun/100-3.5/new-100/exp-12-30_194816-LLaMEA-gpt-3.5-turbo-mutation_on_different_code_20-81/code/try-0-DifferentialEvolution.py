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
        best_idx = np.argmin([func(ind) for ind in population])
        best = population[best_idx]
        
        for i in range(self.budget):
            for j in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
                
                j_rand = np.random.randint(self.dim)
                trial = np.where(np.random.uniform(0, 1, self.dim) < self.CR, mutant, population[j])
                trial[j_rand] = mutant[j_rand]
                
                f_trial = func(trial)
                if f_trial < func(population[j]):
                    population[j] = trial
                    if f_trial < func(best):
                        best = trial
                        if f_trial < self.f_opt:
                            self.f_opt = f_trial
                            self.x_opt = trial
            
        return self.f_opt, self.x_opt