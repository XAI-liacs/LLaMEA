import numpy as np
from scipy.optimize import minimize

class PhotonicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.F_min = 0.5
        self.F_max = 0.9
        self.CR_min = 0.2
        self.CR_max = 0.9
        self.local_budget = int(budget * 0.2)
        self.global_budget = budget - self.local_budget
        self.population = None

    def initialize_population(self, bounds):
        self.population = np.random.uniform(low=bounds.lb, high=bounds.ub, size=(self.population_size, self.dim))
        for i in range(self.population_size):
            self.population[i] = self.population[i] - np.mean(self.population[i]) + np.mean(bounds.lb + bounds.ub) / 2

    def optimize(self, func, bounds):
        evaluations = 0
        self.initialize_population(bounds)
        best_solution = None
        best_score = np.inf

        while evaluations < self.global_budget:
            CR = self.CR_min + (self.CR_max - self.CR_min) * (1 - evaluations / self.global_budget)
            F = self.F_min + (self.F_max - self.F_min) * (evaluations / self.global_budget)
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = self.population[indices]
                if np.random.rand() < 0.5:  # Hybrid mutation strategy
                    mutant = x1 + F * (x2 - x3)
                else:
                    mutant = x1 + F * (best_solution - x1) + F * (x2 - x3)
                mutant = np.clip(mutant, bounds.lb, bounds.ub)
                trial = np.copy(self.population[i])
                for j in range(self.dim):
                    if np.random.rand() < CR:
                        trial[j] = mutant[j]
                trial = self.enforce_periodicity(trial, evaluations / self.global_budget)
                score_trial = func(trial)
                evaluations += 1
                if score_trial < best_score:
                    best_score = score_trial
                    best_solution = trial
                if score_trial < func(self.population[i]):
                    self.population[i] = trial
                if evaluations >= self.global_budget:
                    break

        result = minimize(func, best_solution, bounds=[(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)], method='L-BFGS-B', options={'maxfun': self.local_budget})
        return result.x

    def enforce_periodicity(self, solution, progress):
        period_length = max(2, int((1 - progress) * self.dim / 6))  # Adjusted for adaptive periodicity
        for start in range(0, len(solution), period_length):
            end = start + period_length
            if end <= len(solution):
                avg_value = np.mean(solution[start:end])
                solution[start:end] = avg_value
        return solution

    def __call__(self, func):
        bounds = func.bounds
        return self.optimize(func, bounds)