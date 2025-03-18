import numpy as np
from scipy.optimize import minimize

class HybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.population = None
        self.best_solution = None
        self.best_score = float('-inf')
        self.eval_count = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.best_solution = self.population[0]

    def reflectivity_score(self, x):
        # Simple penalty for non-periodicity (encouraging periodicity)
        period = int(self.dim / 2)  # Assuming a periodic structure
        periodic_deviation = np.sum((x[:period] - x[period:2*period]) ** 2)
        return periodic_deviation

    def differential_evolution(self, func, lb, ub):
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover probability
        
        for i in range(self.population_size):
            if self.eval_count >= self.budget:
                break

            # Mutation and Crossover
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), lb, ub)

            cross_points = np.random.rand(self.dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True

            trial = np.where(cross_points, mutant, self.population[i])

            # Selection
            trial_score = func(trial)
            self.eval_count += 1

            if trial_score > func(self.population[i]) - self.reflectivity_score(trial):
                self.population[i] = trial

            # Update the best solution found
            if trial_score > self.best_score:
                self.best_score = trial_score
                self.best_solution = trial

    def local_search(self, func, lb, ub):
        # Fine-tuning with local optimization
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