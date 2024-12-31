import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        
        if np.min(fitness) < self.f_opt:
            self.f_opt = np.min(fitness)
            self.x_opt = pop[np.argmin(fitness)]
        
        F_min, F_max = 0.4, 1.0
        CR_min, CR_max = 0.1, 0.9
        used_budget = self.pop_size

        while used_budget < self.budget:
            F = np.random.uniform(F_min, F_max)
            CR = np.random.uniform(CR_min, CR_max)
            
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                
                mutant = np.clip(a + F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])
                
                f_trial = func(trial)
                used_budget += 1

                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                if used_budget >= self.budget:
                    break

        return self.f_opt, self.x_opt