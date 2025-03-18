import numpy as np
from scipy.optimize import minimize

class AdaptiveMemoryEnhancedOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 12 * dim
        self.archive_size = int(0.2 * self.population_size)  # Archive for memory-enhanced learning
        self.population = None
        self.archive = []
        self.best_solution = None
        self.best_score = float('-inf')
        self.eval_count = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.best_solution = self.population[0]

    def reflectivity_score(self, x):
        period = self.dim // 2
        periodic_deviation = np.sum((x[:period] - x[period:2*period]) ** 2)
        deviation_penalty = np.sum((x - np.roll(x, period)) ** 2)
        return periodic_deviation + deviation_penalty

    def adaptive_mutation(self, a, b, c, lb, ub):
        F = np.random.uniform(0.5, 0.7)
        mutant = np.clip(a + F * (b - c), lb, ub)
        return mutant

    def coevolutionary_update(self, func, lb, ub):
        for i in range(self.population_size):
            if self.eval_count >= self.budget:
                break

            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
            mutant = self.adaptive_mutation(a, b, c, lb, ub)

            cross_points = np.random.rand(self.dim) < 0.9
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True

            trial = np.where(cross_points, mutant, self.population[i])
            trial_score = func(trial) - self.reflectivity_score(trial)
            self.eval_count += 1

            if trial_score > func(self.population[i]):
                self.population[i] = trial
                if trial_score > self.best_score:
                    self.best_score = trial_score
                    self.best_solution = trial

        self.update_archive()

    def update_archive(self):
        sorted_population = sorted(self.population, key=lambda x: func(x), reverse=True)
        self.archive = sorted_population[:self.archive_size]

    def archive_based_local_search(self, func, lb, ub):
        for solution in self.archive:
            if self.eval_count >= self.budget:
                break

            result = minimize(func, solution, bounds=np.c_[lb, ub], method='L-BFGS-B')
            self.eval_count += result.nfev

            if result.fun > self.best_score:
                self.best_score = result.fun
                self.best_solution = result.x

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)

        while self.eval_count < self.budget:
            self.coevolutionary_update(func, lb, ub)
            self.archive_based_local_search(func, lb, ub)

        return self.best_solution