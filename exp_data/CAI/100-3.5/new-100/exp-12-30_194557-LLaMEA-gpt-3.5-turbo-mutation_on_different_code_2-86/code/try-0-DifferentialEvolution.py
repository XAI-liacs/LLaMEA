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
        pop_size = 10 * self.dim
        bounds = (func.bounds.lb, func.bounds.ub)
        pop = np.random.uniform(bounds[0], bounds[1], (pop_size, self.dim))
        
        for _ in range(self.budget):
            new_pop = []
            for i in range(pop_size):
                a, b, c = np.random.choice(pop, 3, replace=False)
                mutant = a + self.F * (b - c)
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, pop[i])
                
                f_trial = func(trial)
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
                
                new_pop.append(trial if f_trial < func(pop[i]) else pop[i])
                
            pop = np.array(new_pop)
        
        return self.f_opt, self.x_opt