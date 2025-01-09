import numpy as np

class EMOO:
    def __init__(self, budget=10000, dim=10, pop_size=50, f=0.5, cr=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f = f
        self.cr = cr
        self.f_opt = np.Inf
        self.x_opt = None
    
    def local_search(self, x, func):
        delta = 0.01
        for _ in range(10):
            x_new = x + np.random.uniform(-delta, delta, size=self.dim)
            if func(x_new) < func(x):
                x = x_new
        return x
    
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.pop_size, self.dim))
        
        for _ in range(self.budget // self.pop_size):
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = population[a] + self.f * (population[b] - population[c])
                trial = np.where(np.random.rand(self.dim) < self.cr, mutant, population[i])
                
                trial = self.local_search(trial, func)
                
                f = func(trial)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial
        
        return self.f_opt, self.x_opt