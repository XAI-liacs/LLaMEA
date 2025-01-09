import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, NP=50):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.NP = NP
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        def mutate(x, pop, F):
            r1, r2, r3 = np.random.choice(len(pop), 3, replace=False)
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            return np.clip(mutant, -5.0, 5.0)

        def crossover(x, mutant, CR):
            mask = np.random.rand(self.dim) < CR
            trial = np.where(mask, mutant, x)
            return trial

        pop = np.random.uniform(-5.0, 5.0, (self.NP, self.dim))
        for i in range(self.budget):
            for j in range(self.NP):
                x = pop[j]
                mutant = mutate(x, pop, self.F)
                trial = crossover(x, mutant, self.CR)
                f = func(trial)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial
                if f < func(x):
                    pop[j] = trial
        return self.f_opt, self.x_opt