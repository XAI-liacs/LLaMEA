import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=30):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        def ensure_bounds(vec, bounds):
            vec_new = []
            for i in range(len(vec)):
                if vec[i] < bounds.lb:
                    vec_new.append(bounds.lb)
                elif vec[i] > bounds.ub:
                    vec_new.append(bounds.ub)
                else:
                    vec_new.append(vec[i])
            return vec_new

        bounds = func.bounds
        population = np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))

        for i in range(self.budget):
            for j in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                mutant = ensure_bounds(a + self.F * (b - c), bounds)

                crossover = np.random.rand(self.dim) < self.CR
                trial = [mutant[i] if crossover[i] else population[j][i] for i in range(self.dim)]

                f = func(trial)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial
                    population[j] = trial

        return self.f_opt, self.x_opt