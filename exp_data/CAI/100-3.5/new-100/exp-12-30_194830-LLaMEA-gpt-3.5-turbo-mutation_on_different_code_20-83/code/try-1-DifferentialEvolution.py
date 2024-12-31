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
        pop_size = 10*self.dim
        pop = np.random.uniform(-5.0, 5.0, size=(pop_size, self.dim))
        
        for i in range(self.budget):
            F_mutant = np.clip(np.random.normal(self.F, 0.1), 0.1, 1.0) # Adaptive mutation factor
            CR_crossover = np.clip(np.random.normal(self.CR, 0.1), 0.1, 1.0) # Adaptive crossover rate
            
            for j in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F_mutant*(b - c), -5.0, 5.0)
                crossover = np.random.rand(self.dim) < CR_crossover
                trial = np.where(crossover, mutant, pop[j])
                
                f_trial = func(trial)
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
                    pop[j] = trial
            
        return self.f_opt, self.x_opt