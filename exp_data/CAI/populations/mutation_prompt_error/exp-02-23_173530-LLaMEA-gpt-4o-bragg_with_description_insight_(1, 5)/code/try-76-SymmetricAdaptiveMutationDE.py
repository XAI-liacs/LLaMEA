import numpy as np
from scipy.optimize import minimize

class SymmetricAdaptiveMutationDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def _reflectivity_cost(self, params, func):
        if self.eval_count < self.budget:
            self.eval_count += 1
            return func(params)
        else:
            raise RuntimeError("Exceeded budget!")

    def _symmetric_initialization(self, bounds, pop_size):
        """ Initialize population using symmetric sampling """
        lb, ub = bounds.lb, bounds.ub
        mid_point = (lb + ub) / 2
        radius = (ub - lb) / 2
        population = mid_point + (np.random.rand(pop_size, self.dim) - 0.5) * 2 * radius
        return np.clip(population, lb, ub)

    def _adaptive_periodic_mutation(self, individual, F, bounds):
        """ Apply adaptive mutation with periodicity to individuals """
        period = 3 + int((self.eval_count / self.budget) * (self.dim // 3))
        lb, ub = bounds.lb, bounds.ub
        for i in range(0, self.dim, period):
            end = min(i + period, self.dim)
            noise = F * (np.random.rand(end - i) - 0.5) * (ub[i:end] - lb[i:end])
            individual[i:end] = np.clip(individual[i:end] + noise, lb[i:end], ub[i:end])
        return individual

    def _differential_evolution(self, func, bounds, pop_size=20, max_iter=100):
        """ Differential Evolution with symmetric initialization and adaptive periodic mutation """
        population = self._symmetric_initialization(bounds, pop_size)
        costs = np.array([self._reflectivity_cost(ind, func) for ind in population])
        best_idx = np.argmin(costs)
        best = population[best_idx].copy()

        for gen in range(max_iter):
            F = 0.5 + 0.3 * np.cos(gen / max_iter * np.pi)
            CR = 0.6 + 0.2 * np.sin(gen / max_iter * np.pi/2)
            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = self._adaptive_periodic_mutation(a + F * (b - c), F, bounds)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, population[i])
                cost_trial = self._reflectivity_cost(trial, func)
                if cost_trial < costs[i]:
                    population[i] = trial
                    costs[i] = cost_trial
                    if cost_trial < self._reflectivity_cost(best, func):
                        best = trial.copy()

        return best

    def __call__(self, func):
        bounds = func.bounds
        best_global = self._differential_evolution(func, bounds)
        result = minimize(lambda x: self._reflectivity_cost(x, func), best_global, method='L-BFGS-B', bounds=[(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)])

        return result.x if result.success else best_global