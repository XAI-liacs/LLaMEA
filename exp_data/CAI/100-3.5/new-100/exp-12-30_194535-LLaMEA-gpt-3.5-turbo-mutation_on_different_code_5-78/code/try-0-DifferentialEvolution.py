import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, f=0.5, cr=0.9):
        self.budget = budget
        self.dim = dim
        self.f = f
        self.cr = cr
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop_size = 10 * self.dim
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = np.random.uniform(bounds[0], bounds[1], (pop_size, self.dim))
        
        for _ in range(self.budget // pop_size):
            for i in range(pop_size):
                target = population[i]
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), bounds[0], bounds[1])
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, target)
                
                f_target = func(target)
                f_trial = func(trial)
                
                if f_trial < f_target:
                    population[i] = trial
                    
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        
        return self.f_opt, self.x_opt