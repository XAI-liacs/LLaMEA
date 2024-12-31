import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # Scaling factor
        self.CR = CR  # Crossover probability
        self.f_opt = np.Inf
        self.x_opt = None
        self.pop = None
        self.func_evals = 0

    def initialize_population(self, bounds):
        self.pop = np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))

    def mutate(self, idx):
        candidates = list(range(self.pop_size))
        candidates.remove(idx)
        a, b, c = np.random.choice(candidates, 3, replace=False)
        donor = self.pop[a] + self.F * (self.pop[b] - self.pop[c])
        return np.clip(donor, -5.0, 5.0)

    def crossover(self, target, donor):
        mask = np.random.rand(self.dim) < self.CR
        trial = np.where(mask, donor, target)
        return trial

    def select(self, target_idx, trial, func):
        f_target = func(self.pop[target_idx])
        f_trial = func(trial)
        self.func_evals += 2  # We evaluate the function twice per selection
        if f_trial < f_target:
            self.pop[target_idx] = trial
            if f_trial < self.f_opt:
                self.f_opt = f_trial
                self.x_opt = trial

    def evolve(self, func):
        while self.func_evals < self.budget:
            for idx in range(self.pop_size):
                donor = self.mutate(idx)
                trial = self.crossover(self.pop[idx], donor)
                self.select(idx, trial, func)
                if self.func_evals >= self.budget:
                    break

    def __call__(self, func):
        self.initialize_population(func.bounds)
        self.func_evals = 0
        self.evolve(func)
        return self.f_opt, self.x_opt