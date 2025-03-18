import numpy as np
from scipy.optimize import minimize

class PhotonicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.used_budget = 0

    def symmetric_initialization(self, lb, ub, pop_size):
        mid = (ub + lb) / 2
        range_span = (ub - lb)
        population = np.random.uniform(lb, ub, (pop_size, self.dim))
        symmetric_population = 2 * mid - population
        return np.vstack((population, symmetric_population))

    def apply_periodicity(self, solution):
        period = self.dim // 4
        num_repeats = len(solution) // period
        base_pattern = np.fft.ifft(np.fft.fft(solution[:period])).real
        for i in range(1, num_repeats):
            solution[i*period:(i+1)*period] = base_pattern
        return solution

    def differential_evolution(self, func, bounds, pop_size=20, F=0.5, CR=0.9):
        lb, ub = bounds.lb, bounds.ub
        dynamic_pop_size = int(pop_size * (1 + 0.15 * (self.used_budget / self.budget)))
        population = self.symmetric_initialization(lb, ub, dynamic_pop_size)
        fitness = np.apply_along_axis(func, 1, population)
        self.used_budget += len(fitness)

        while self.used_budget < self.budget:
            for i in range(dynamic_pop_size):
                if self.used_budget >= self.budget:
                    break

                idxs = [idx for idx in range(dynamic_pop_size * 2) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                F = np.random.uniform(0.4, 1.0) * (1 + 0.1 * (self.used_budget / self.budget))  # Changed mutation scaling
                mutant = np.clip(a + F * (b - c), lb, ub)
                diversity_factor = np.std(population) / np.mean(population)
                CR = 0.85 - 0.4 * (self.used_budget / self.budget) * (1 + 0.4 * diversity_factor)  # Adjusted CR dynamics
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial = self.apply_periodicity(trial)
                f = func(trial)
                self.used_budget += 1
                if f < fitness[i]:
                    fitness[i] = f
                    population[i] = trial

            # Adaptive learning rate for diversity
            population_variance = np.var(population, axis=0)
            if np.max(population_variance) < 1e-3:
                additional_mutant = np.random.normal(0, 0.01, self.dim)
                population += additional_mutant

        return population[np.argmin(fitness)]

    def local_optimization(self, func, x0):
        if self.used_budget < self.budget:
            result = minimize(lambda x: func(x) + 0.05 * np.sum(np.sin(2 * np.pi * x)), x0 + np.random.normal(0, 0.001, self.dim),  # Introduced Gaussian mutation
                              method='L-BFGS-B', bounds=list(zip(func.bounds.lb, func.bounds.ub)))
            self.used_budget += result.nfev
            return result.x
        return x0

    def __call__(self, func):
        bounds = func.bounds
        best_solution = self.differential_evolution(func, bounds)
        best_solution = self.local_optimization(func, best_solution)
        return best_solution