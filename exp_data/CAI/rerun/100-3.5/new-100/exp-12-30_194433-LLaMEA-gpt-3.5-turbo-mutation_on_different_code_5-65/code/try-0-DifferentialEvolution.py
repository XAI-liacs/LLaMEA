import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def mutate(self, pop, target_idx):
        candidates = [idx for idx in range(len(pop)) if idx != target_idx]
        a, b, c = np.random.choice(candidates, 3, replace=False)
        mutant = pop[a] + self.F * (pop[b] - pop[c])
        return np.clip(mutant, -5.0, 5.0)

    def __call__(self, func):
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        for _ in range(self.budget // self.pop_size):
            for i in range(self.pop_size):
                x = pop[i]
                mutant = self.mutate(pop, i)
                crossover = np.random.rand(self.dim) <= self.CR
                trial = np.where(crossover, mutant, x)
                f = func(trial)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial
                pop[i] = trial if f < func(x) else x
                
        return self.f_opt, self.x_opt