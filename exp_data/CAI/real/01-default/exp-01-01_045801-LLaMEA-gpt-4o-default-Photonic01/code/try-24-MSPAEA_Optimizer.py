import numpy as np

class MSPAEA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 30
        self.population_size = self.initial_population_size
        self.population = None
        self.scores = None
        self.best_solution = None
        self.best_score = float('inf')
        self.evaluations = 0
        self.F = 0.5  # Initial mutation factor
        self.CR = 0.9  # Initial crossover probability
        self.mutation_strategies = [self.rand_1, self.rand_2, self.best_1]

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.scores = np.array([self.evaluate(ind) for ind in self.population])
        best_idx = np.argmin(self.scores)
        self.best_solution = self.population[best_idx].copy()
        self.best_score = self.scores[best_idx]

    def evaluate(self, solution):
        return self.func(solution)

    def adjust_parameters(self):
        self.F = 0.4 + 0.6 * (self.budget - self.evaluations) / self.budget
        self.CR = 0.6 + 0.4 * (self.evaluations / self.budget)

    def rand_1(self, target_idx):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        return a + self.F * (b - c)

    def rand_2(self, target_idx):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c, d, e = self.population[np.random.choice(indices, 5, replace=False)]
        return a + self.F * (b - c) + self.F * (d - e)

    def best_1(self, target_idx):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b = self.population[np.random.choice(indices, 2, replace=False)]
        return self.best_solution + self.F * (a - b)

    def mutate(self, target_idx):
        strategy = np.random.choice(self.mutation_strategies)
        return strategy(target_idx)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        return np.where(crossover_mask, mutant, target)

    def select(self, target_idx, trial):
        trial_score = self.evaluate(trial)
        if trial_score < self.scores[target_idx]:
            self.population[target_idx] = trial
            self.scores[target_idx] = trial_score
            if trial_score < self.best_score:
                self.best_score = trial_score
                self.best_solution = trial.copy()

    def resize_population(self):
        self.population_size = max(5, int(self.initial_population_size * (1 - 0.8 * self.evaluations / self.budget)))
        self.population = self.population[:self.population_size]
        self.scores = self.scores[:self.population_size]

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            self.resize_population()
            self.adjust_parameters()
            for i in range(self.population_size):
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                trial = np.clip(trial, lb, ub)
                self.select(i, trial)
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break

        return {'solution': self.best_solution, 'fitness': self.best_score}