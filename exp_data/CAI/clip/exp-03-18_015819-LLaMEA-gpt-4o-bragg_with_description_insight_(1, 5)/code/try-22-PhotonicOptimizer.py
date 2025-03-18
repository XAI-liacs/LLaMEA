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
        period = self.dim // 2
        for i in range(0, self.dim, period):
            solution[i:i+period] = solution[:period]
        return solution

    def differential_evolution(self, func, bounds, pop_size=20, F=0.5, CR=0.9):
        lb, ub = bounds.lb, bounds.ub
        population = self.symmetric_initialization(lb, ub, pop_size)
        fitness = np.apply_along_axis(func, 1, population)
        self.used_budget += len(fitness)

        while self.used_budget < self.budget:
            for i in range(pop_size):
                if self.used_budget >= self.budget:
                    break

                idxs = [idx for idx in range(pop_size * 2) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                F = np.random.uniform(0.3, 0.9)  # Introduce stochastic weighting for F
                mutant = a + F * (b - c)
                mutant = np.clip(mutant, lb + 0.1 * (ub - lb), ub - 0.1 * (ub - lb))  # Adaptive boundary handling
                CR = 0.8 - 0.4 * (self.used_budget / self.budget)  # Changed Adaptive CR
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial = self.apply_periodicity(trial)  # Enforce periodicity
                f = func(trial)
                self.used_budget += 1
                if f < fitness[i]:
                    fitness[i] = f
                    population[i] = trial

        return population[np.argmin(fitness)]

    def local_optimization(self, func, x0):
        if self.used_budget < self.budget:
            result = minimize(lambda x: func(x) + 0.05 * np.sum(np.sin(2 * np.pi * x)), x0, 
                              method='L-BFGS-B', bounds=list(zip(func.bounds.lb, func.bounds.ub)))  # Refined sinusoidal perturbation
            self.used_budget += result.nfev
            return result.x
        return x0

    def __call__(self, func):
        bounds = func.bounds
        best_solution = self.differential_evolution(func, bounds)
        best_solution = self.local_optimization(func, best_solution)
        return best_solution