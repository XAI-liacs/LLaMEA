import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def differential_evolution(self, func, bounds, pop_size=15, F=0.8, Cr=0.9):
        population = np.random.rand(pop_size, self.dim) * (bounds.ub - bounds.lb) + bounds.lb
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += pop_size

        while self.eval_count < self.budget:
            for i in range(pop_size):
                if self.eval_count >= self.budget:
                    break
                indices = list(range(pop_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
                trial = np.where(np.random.rand(self.dim) < Cr, mutant, population[i])
                trial_fitness = func(trial)
                self.eval_count += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]

    def local_optimization(self, func, x0):
        x0 = x0 + np.random.normal(0, 0.01, size=self.dim)  # Slightly perturb initial point
        result = minimize(func, x0, method='BFGS', options={'maxiter': self.budget - self.eval_count})
        self.eval_count += result.nfev
        return result.x, result.fun

    def enforce_periodicity(self, x, period):
        return np.tile(x[:period], self.dim // period)

    def __call__(self, func):
        bounds = func.bounds
        period = self.dim // 2  # Aim for a periodic solution with half the dimensions
        x_global, f_global = self.differential_evolution(func, bounds)

        x_periodic = self.enforce_periodicity(x_global, period)
        x_local, f_local = self.local_optimization(func, x_periodic)

        return x_local if f_local < f_global else x_global