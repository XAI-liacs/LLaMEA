import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, f=0.5, cr=0.9, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.f = f
        self.cr = cr
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        
        for _ in range(self.budget // self.pop_size):
            for i in range(self.pop_size):
                target = population[i]
                
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                mutant = population[a] + self.f * (population[b] - population[c])
                crossover_points = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover_points, mutant, target)
                
                f_target = func(target)
                f_trial = func(trial)
                
                if f_trial < f_target:
                    population[i] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
        
        return self.f_opt, self.x_opt