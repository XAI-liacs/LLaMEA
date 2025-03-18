import numpy as np
from scipy.optimize import minimize

class EnhancedHybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 12 * dim
        self.population = None
        self.best_solution = None
        self.best_score = float('-inf')
        self.eval_count = 0
        self.enhanced_periodicity_weight = 0.1  # New periodicity weight

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.best_solution = self.population[0]

    def reflectivity_score(self, x):
        period = np.random.randint(1, self.dim // 2)
        periodic_deviation = np.sum((x[:period] - x[period:2*period]) ** 2)
        # Added enhanced periodicity score
        return periodic_deviation + self.enhanced_periodicity_weight * np.std(x)

    def differential_evolution(self, func, lb, ub):
        F = np.random.uniform(0.5, 0.9)
        CR = np.random.uniform(0.8, 1.0)
        
        for i in range(self.population_size):
            if self.eval_count >= self.budget:
                break

            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), lb, ub)

            cross_points = np.random.rand(self.dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True

            trial = np.where(cross_points, mutant, self.population[i])

            trial_score = func(trial)
            self.eval_count += 1

            if trial_score > func(self.population[i]) - self.reflectivity_score(trial):
                self.population[i] = trial

            if trial_score > self.best_score:
                self.best_score = trial_score
                self.best_solution = trial

    def stochastic_local_search(self, func, lb, ub):
        # Using a stochastic search method
        perturbation = np.random.uniform(-0.05, 0.05, self.dim)
        candidate = self.best_solution + perturbation
        candidate = np.clip(candidate, lb, ub)

        candidate_score = func(candidate)
        self.eval_count += 1

        if candidate_score > self.best_score:
            self.best_score = candidate_score
            self.best_solution = candidate

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)

        while self.eval_count < self.budget:
            self.differential_evolution(func, lb, ub)
            self.stochastic_local_search(func, lb, ub)

        return self.best_solution