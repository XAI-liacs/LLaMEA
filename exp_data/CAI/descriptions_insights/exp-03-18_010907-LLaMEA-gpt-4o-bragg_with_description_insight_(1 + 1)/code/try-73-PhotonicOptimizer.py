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
        quasi_opposite_population = bounds.lb + bounds.ub - self.population
        self.population = np.vstack((self.population, quasi_opposite_population))[:self.population_size]
        for i in range(self.population_size):
            self.population[i] = self.population[i] - np.mean(self.population[i]) + np.mean(bounds.lb + bounds.ub) / 2

    def optimize(self, func, bounds):
        evaluations = 0
        self.initialize_population(bounds)
        best_solution = None
        best_score = np.inf

        while evaluations < self.global_budget:
            CR = self.CR_min + (self.CR_max - self.CR_min) * np.random.rand()  # Adaptive CR
            factor = (evaluations / self.global_budget) ** 1.2  # Enhanced non-linear scaling
            F = self.F_min + (self.F_max - self.F_min) * factor  # Enhanced non-linear scaling
            perturbation_variance = 0.05 * np.exp(-evaluations / self.global_budget)  # Refined perturbation variance decay
            self.population_size = max(10, int(20 * (1 - factor)))  # Adaptive population resizing
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = self.population[indices]
                mutant = x1 + F * (x2 - x3) + np.random.normal(0, perturbation_variance, size=self.dim)  # Adaptive perturbation
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
        scale_factor = 0.6 + 0.35 * np.cos(progress * np.pi / 2 + 0.1)  # Adjusted dynamic scaling
        phase_shift = 0.55 * np.sin(progress * np.pi / 2 + 0.1) + 0.05  # Adjusted phase shift for periodicity
        period_length = max(1, int((1 - scale_factor * np.sin(progress * np.pi / 2 + phase_shift)) * self.dim / 4))  # Adjusted for dynamic scaling
        for start in range(0, len(solution), period_length):
            end = start + period_length
            if end <= len(solution):
                avg_value = np.mean(solution[start:end])
                solution[start:end] = avg_value
        return solution

    def __call__(self, func):
        bounds = func.bounds
        return self.optimize(func, bounds)