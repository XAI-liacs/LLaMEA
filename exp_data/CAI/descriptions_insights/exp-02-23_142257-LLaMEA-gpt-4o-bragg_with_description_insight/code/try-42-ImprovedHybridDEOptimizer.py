import numpy as np
from scipy.optimize import minimize

class ImprovedHybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 12 * dim
        self.population = None
        self.best_solution = None
        self.best_score = float('-inf')
        self.eval_count = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.initial_population_size, self.dim))
        self.best_solution = self.population[0]

    def differential_evolution(self, func, lb, ub):
        F = np.random.uniform(0.5, 0.9)  # Adaptive differential weight
        CR = np.random.uniform(0.7, 0.95)  # Adjusted crossover probability
        dynamic_population_size = max(4, int(self.initial_population_size * (1 - self.eval_count / self.budget)))

        for i in range(dynamic_population_size):
            if self.eval_count >= self.budget:
                break

            idxs = [idx for idx in range(dynamic_population_size) if idx != i]
            a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), lb, ub)

            cross_points = np.random.rand(self.dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True

            trial = np.where(cross_points, mutant, self.population[i])
            trial_score = func(trial)
            self.eval_count += 1

            periodic_score = self.enforce_periodicity(trial)

            if trial_score + periodic_score > func(self.population[i]):
                self.population[i] = trial

            if trial_score > self.best_score:
                self.best_score = trial_score
                self.best_solution = trial

    def enforce_periodicity(self, solution):
        period = self.dim // 2
        periodicity_penalty = np.sum((solution[:period] - solution[period:period*2]) ** 2)
        return -periodicity_penalty

    def local_search(self, func, lb, ub):
        result = minimize(func, self.best_solution, bounds=np.c_[lb, ub], method='L-BFGS-B')
        self.eval_count += result.nfev

        if result.fun > self.best_score:
            self.best_score = result.fun
            self.best_solution = result.x

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)

        while self.eval_count < self.budget:
            self.differential_evolution(func, lb, ub)
            self.local_search(func, lb, ub)

        return self.best_solution