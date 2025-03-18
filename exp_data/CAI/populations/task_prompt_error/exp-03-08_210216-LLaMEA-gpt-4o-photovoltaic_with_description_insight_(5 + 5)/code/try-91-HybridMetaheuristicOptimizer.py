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
        for i in range(self.pop_size):
            if self.evaluations >= self.budget:
                break
            idxs = [idx for idx in range(self.pop_size) if idx != i]
            a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
            self.F = 0.5 + 0.3 * np.std(self.population)  # Adapt mutation factor
            self.CR = np.clip(0.5 + 0.3 * np.std(self.population), 0, 1)  # Adapt crossover probability
            mutant = np.clip(a + self.F * (b - c), 0, 1)
            crossover = np.random.rand(self.dim) < self.CR
            trial = np.where(crossover, mutant, self.population[i])
            trial_denorm = bounds.lb + trial * (bounds.ub - bounds.lb)
            score = func(trial_denorm)
            self.evaluations += 1
            if score < self.scores[i]:
                self.population[i] = trial
                self.scores[i] = score

    def exploitative_refinement(self, func, individual, bounds):
        individual_denorm = bounds.lb + individual * (bounds.ub - bounds.lb)
        result = minimize(func, individual_denorm, method='L-BFGS-B', bounds=list(zip(bounds.lb, bounds.ub)))
        if result.success:
            return result.x, result.fun
        else:
            return individual_denorm, func(individual_denorm)

    def resize_population(self, factor):
        new_size = max(10, int(self.pop_size * factor))
        self.population = np.random.rand(new_size, self.dim)
        self.scores = np.full(new_size, np.inf)
        self.pop_size = new_size

    def __call__(self, func):
        bounds = func.bounds
        target_dim = bounds.ub.size
        resize_factors = [1.0, 0.8, 1.2]  # Different factors to dynamically resize population
        resize_idx = 0
        while self.evaluations < self.budget:
            if self.evaluations % (self.budget // 10) == 0 and resize_idx < len(resize_factors):
                self.resize_population(resize_factors[resize_idx])
                resize_idx += 1
            self.evaluate_population(func)
            self.differential_evolution_step(func, bounds)
            best_idx = np.argmin(self.scores)
            best_individual = self.population[best_idx]
            if self.evaluations < self.budget:
                refined_solution, refined_score = self.exploitative_refinement(func, best_individual, bounds)
                if refined_score < self.scores[best_idx]:
                    self.population[best_idx] = (refined_solution - bounds.lb) / (bounds.ub - bounds.lb)
                    self.scores[best_idx] = refined_score

        best_idx = np.argmin(self.scores)
        best_solution = bounds.lb + self.population[best_idx] * (bounds.ub - bounds.lb)
        return best_solution, self.scores[best_idx]