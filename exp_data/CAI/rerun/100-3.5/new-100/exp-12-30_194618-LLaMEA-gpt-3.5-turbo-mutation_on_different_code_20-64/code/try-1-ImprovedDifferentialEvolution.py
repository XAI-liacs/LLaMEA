import numpy as np

class ImprovedDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.8, CR=0.9, pop_size=50, F_lower=0.2, F_upper=0.9, F_decay=0.99):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.F_lower = F_lower
        self.F_upper = F_upper
        self.F_decay = F_decay
        self.f_opt = np.Inf
        self.x_opt = None
        self.iteration = 0
    
    def __call__(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        
        for _ in range(self.budget):
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
                
                j_rand = np.random.randint(self.dim)
                trial = np.copy(pop[i])
                trial[j_rand] = mutant[j_rand]

                f_trial = func(trial)
                if f_trial < func(pop[i]):
                    pop[i] = trial

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
            
            self.iteration += 1
            self.F = max(self.F_lower, self.F * self.F_decay) if self.iteration % 10 == 0 else self.F  # Adjust F every 10 iterations
            
        return self.f_opt, self.x_opt