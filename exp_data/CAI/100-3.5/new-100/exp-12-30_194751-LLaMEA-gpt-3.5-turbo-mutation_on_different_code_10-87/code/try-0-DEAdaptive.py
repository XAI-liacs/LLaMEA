import numpy as np

class DEAdaptive:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.pop_size = 10  # Initial population size
        self.cr = 0.9  # Crossover probability
        self.f = 0.8  # Differential weight
        self.pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))

    def __call__(self, func):
        for i in range(self.budget):
            for j in range(self.pop_size):
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = self.pop[idxs]
                mutant = a + self.f * (b - c)
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, self.pop[j])
                
                f = func(trial)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial
                if f < func(self.pop[j]):
                    self.pop[j] = trial

            # Adaptive strategy for population size
            self.pop_size = int(5 + 15 * (i / self.budget))
            self.pop = np.vstack([self.pop, np.random.uniform(-5.0, 5.0, (self.pop_size - len(self.pop), self.dim))])

        return self.f_opt, self.x_opt