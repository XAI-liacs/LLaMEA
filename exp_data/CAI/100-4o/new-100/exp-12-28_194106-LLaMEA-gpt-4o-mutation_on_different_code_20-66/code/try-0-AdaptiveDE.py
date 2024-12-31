import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.population = None
        self.func_evals = 0
        self.f_opt = np.Inf
        self.x_opt = None

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))

    def mutate(self, target_idx):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        return self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])

    def crossover(self, target, donor):
        trial = np.copy(target)
        for j in range(self.dim):
            if np.random.rand() < self.crossover_rate or j == np.random.randint(self.dim):
                trial[j] = donor[j]
        return trial

    def select(self, target_idx, trial, func):
        target = self.population[target_idx]
        f_target = func(target)
        f_trial = func(trial)
        self.func_evals += 2  # count evaluations for both target and trial

        if f_trial < f_target:
            self.population[target_idx] = trial
            if f_trial < self.f_opt:
                self.f_opt = f_trial
                self.x_opt = trial
        else:
            if f_target < self.f_opt:
                self.f_opt = f_target
                self.x_opt = target

    def adapt_parameters(self):
        if np.random.rand() > 0.5:
            self.mutation_factor = np.clip(self.mutation_factor + np.random.normal(0, 0.1), 0.1, 1.0)
            self.crossover_rate = np.clip(self.crossover_rate + np.random.normal(0, 0.1), 0.1, 1.0)

    def __call__(self, func):
        self.initialize_population(func.bounds.lb, func.bounds.ub)

        while self.func_evals < self.budget:
            for i in range(self.population_size):
                if self.func_evals >= self.budget:
                    break
                donor = self.mutate(i)
                trial = self.crossover(self.population[i], donor)
                self.select(i, trial, func)
                self.adapt_parameters()

        return self.f_opt, self.x_opt