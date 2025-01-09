import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=20):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.population = None
        self.func_evals = 0

    def initialize(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.evaluate_population()

    def evaluate_population(self):
        for i in range(self.population_size):
            f = self.evaluate(self.population[i])
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = self.population[i]

    def evaluate(self, x):
        f = self.func(x)
        self.func_evals += 1
        return f

    def mutate(self, target_idx):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        return self.population[a] + self.F * (self.population[b] - self.population[c])

    def crossover(self, target, mutant):
        j_rand = np.random.randint(0, self.dim)
        trial = np.copy(target)
        for j in range(self.dim):
            if np.random.rand() < self.CR or j == j_rand:
                trial[j] = mutant[j]
        return trial

    def select(self, target_idx, trial):
        target = self.population[target_idx]
        f_target = self.evaluate(target)
        f_trial = self.evaluate(trial)
        if f_trial < f_target:
            self.population[target_idx] = trial
            if f_trial < self.f_opt:
                self.f_opt = f_trial
                self.x_opt = trial

    def adapt_parameters(self):
        self.F = np.random.normal(0.8, 0.1)  # Gaussian distribution for F
        self.CR = np.random.beta(0.5, 0.5)  # Beta distribution for CR

    def __call__(self, func):
        self.func = func
        self.func_evals = 0
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize(lb, ub)

        while self.func_evals < self.budget:
            self.adapt_parameters()
            for i in range(self.population_size):
                if self.func_evals >= self.budget:
                    break
                mutant = self.mutate(i)
                mutant = np.clip(mutant, lb, ub)
                trial = self.crossover(self.population[i], mutant)
                self.select(i, trial)

        return self.f_opt, self.x_opt