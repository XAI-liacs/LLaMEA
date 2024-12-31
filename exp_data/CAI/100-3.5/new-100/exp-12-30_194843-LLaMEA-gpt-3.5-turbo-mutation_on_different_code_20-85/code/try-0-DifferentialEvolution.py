import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, npop=50, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.npop = npop
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None
    
    def __call__(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.npop, self.dim))
        
        for i in range(self.budget):
            for j in range(self.npop):
                idxs = [idx for idx in range(self.npop) if idx != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
                
                j_rand = np.random.randint(self.dim)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, pop[j])
                trial[j_rand] = mutant[j_rand]  # Ensure at least one parameter change
                
                f = func(trial)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial.copy()
                    
                if i == self.budget - 1:
                    break
                
                if f < func(pop[j]):
                    pop[j] = trial
        
        return self.f_opt, self.x_opt