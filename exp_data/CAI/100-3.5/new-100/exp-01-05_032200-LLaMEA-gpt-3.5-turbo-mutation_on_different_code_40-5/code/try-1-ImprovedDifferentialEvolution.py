import numpy as np

class ImprovedDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        cr = 0.5 + 0.3 * np.random.rand()
        f = 0.5 + 0.3 * np.random.rand()
        for i in range(self.budget):
            for j in range(self.pop_size):
                idxs = list(range(self.pop_size))
                idxs.remove(j)
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = pop[a] + f * (pop[b] - pop[c])
                crossover = np.random.rand(self.dim) < cr
                trial = np.where(crossover, mutant, pop[j])
                
                f_trial = func(trial)
                if f_trial < func(pop[j]):
                    pop[j] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

        return self.f_opt, self.x_opt