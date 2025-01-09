import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=None):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size if pop_size else 8 * dim  # Slightly reduced initial population size
        self.f_opt = np.Inf
        self.x_opt = None
        self.population = None
        self.func_evals = 0

    def initialize_population(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))

    def adaptive_mutate(self, idx):
        candidates = [i for i in range(self.pop_size) if i != idx]
        a, b, c = np.random.choice(candidates, 3, replace=False)
        F = np.random.uniform(0.4, 0.9) + np.std(self.population) * 0.1  # Dynamic scaling factor based on diversity
        return self.population[a] + F * (self.population[b] - self.population[c])

    def crossover(self, target, mutant):
        CR = np.random.uniform(0.2, 0.8)  # Slightly adjusted crossover rate range
        crossover_mask = np.random.rand(self.dim) < CR
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def select(self, target_idx, trial, func):
        target = self.population[target_idx]
        f_target = func(target)
        self.func_evals += 1

        f_trial = func(trial)
        self.func_evals += 1

        if f_trial < f_target:
            self.population[target_idx] = trial
            if f_trial < self.f_opt:
                self.f_opt = f_trial
                self.x_opt = trial

        # Adaptive population adjustment
        if self.func_evals % (self.budget // 5) == 0:  # Every 20% of the budget
            self.pop_size = max(4, int(0.9 * self.pop_size))  # Reduce population to intensify search

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)
        while self.func_evals < self.budget:
            for i in range(self.pop_size):
                if self.func_evals >= self.budget:
                    break
                mutant = self.adaptive_mutate(i)
                trial = self.crossover(self.population[i], mutant)
                trial = np.clip(trial, bounds.lb, bounds.ub)  # Ensure bounds
                self.select(i, trial, func)

        return self.f_opt, self.x_opt