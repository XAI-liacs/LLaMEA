import numpy as np

class HADE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.F = 0.5  # Mutation factor
        self.CR = 0.9  # Crossover probability
        self.population = None
        self.scores = None
        self.best_solution = None
        self.best_score = float('inf')

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.scores = np.array([self.evaluate(ind) for ind in self.population])
        best_idx = np.argmin(self.scores)
        self.best_solution = self.population[best_idx].copy()
        self.best_score = self.scores[best_idx]

    def evaluate(self, solution):
        return self.func(solution)

    def mutate(self, target_idx):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        mutant = a + self.F * (b - c)
        return mutant

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

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub
        evaluations = 0

        self.initialize_population(lb, ub)

        while evaluations < self.budget:
            for i in range(self.population_size):
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                trial = np.clip(trial, lb, ub)
                self.select(i, trial)
                evaluations += 1
                if evaluations >= self.budget:
                    break

        return {'solution': self.best_solution, 'fitness': self.best_score}