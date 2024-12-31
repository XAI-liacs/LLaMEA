import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F_init=0.5, CR_init=0.9, F_decay=0.9, CR_increase=0.1):
        self.budget = budget
        self.dim = dim
        self.F = F_init
        self.CR = CR_init
        self.F_decay = F_decay
        self.CR_increase = CR_increase
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop_size = 10 * self.dim
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(pop_size, self.dim))
        
        for i in range(self.budget):
            for j in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = population[a] + self.F * (population[b] - population[c])
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[j])
                
                f_trial = func(trial)
                if f_trial < func(population[j]):
                    population[j] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
            
            self.F *= self.F_decay
            self.CR += self.CR_increase if np.random.rand() < 0.1 else 0
            
        return self.f_opt, self.x_opt