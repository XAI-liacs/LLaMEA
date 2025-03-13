import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.pop_size = min(100, 10 * dim)
        self.F = 0.8  # DE mutation factor
        self.CR = 0.9  # DE crossover probability
        self.population = np.random.rand(self.pop_size, dim)
        self.scores = np.full(self.pop_size, np.inf)

    def evaluate_population(self, func):
        for i in range(self.pop_size):
            if self.scores[i] == np.inf:
                self.scores[i] = func(self.population[i])
                self.evaluations += 1

    def differential_evolution_step(self, func, bounds):
        diversity = np.std(self.population)
        for i in range(self.pop_size):
            if self.evaluations >= self.budget:
                break
            idxs = [idx for idx in range(self.pop_size) if idx != i]
            a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
            improvement = (self.scores[i] - np.min(self.scores)) / (np.ptp(self.scores) + 1e-9)
            self.F = 0.6 + 0.3 * improvement  # Adapt mutation factor based on improvement
            self.CR = np.clip(0.6 + 0.3 * improvement, 0, 1)  # Adapt crossover probability
            mutant = np.clip(a + self.F * (b - c) + 0.01 * np.random.randn(*a.shape), 0, 1)
            crossover = np.random.rand(self.dim) < self.CR
            trial = np.where(crossover, mutant, self.population[i])
            trial_denorm = bounds.lb + trial * (bounds.ub - bounds.lb)
            score = func(trial_denorm)
            self.evaluations += 1
            if score < self.scores[i]:
                self.population[i] = trial
                self.scores[i] = score

    def local_search(self, func, individual, bounds):
        individual_denorm = bounds.lb + individual * (bounds.ub - bounds.lb)
        result = minimize(func, individual_denorm, method='L-BFGS-B', bounds=list(zip(bounds.lb, bounds.ub)))
        if result.success:
            return result.x, result.fun
        else:
            return individual_denorm, func(individual_denorm)

    def detect_modular_structure(self):
        return

    def adapt_dimensionality(self, current_dim, target_dim):
        if current_dim < target_dim:
            new_dim = current_dim + 2  # Faster dimensionality increase
        else:
            new_dim = current_dim
        return new_dim

    def __call__(self, func):
        bounds = func.bounds
        target_dim = bounds.ub.size
        while self.evaluations < self.budget:
            self.evaluate_population(func)
            self.differential_evolution_step(func, bounds)
            best_idx = np.argmin(self.scores)
            best_individual = self.population[best_idx]
            if self.evaluations < self.budget and (self.evaluations % 10 == 0):
                refined_solution, refined_score = self.local_search(func, best_individual, bounds)
                if refined_score < self.scores[best_idx]:
                    self.population[best_idx] = (refined_solution - bounds.lb) / (bounds.ub - bounds.lb)
                    self.scores[best_idx] = refined_score
            self.dim = self.adapt_dimensionality(self.dim, target_dim)
            
        best_idx = np.argmin(self.scores)
        best_solution = bounds.lb + self.population[best_idx] * (bounds.ub - bounds.lb)
        return best_solution, self.scores[best_idx]