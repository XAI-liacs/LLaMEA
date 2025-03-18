import numpy as np
from scipy.optimize import minimize

class CrowdingDEOptimizer:
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

    def _crowding_distance(self, population):
        """ Calculate crowding distance to maintain diversity """
        pop_size = population.shape[0]
        distances = np.zeros(pop_size)
        for dim in range(self.dim):
            sorted_indices = np.argsort(population[:, dim])
            sorted_population = population[sorted_indices]
            distances[sorted_indices[0]] = distances[sorted_indices[-1]] = np.inf
            min_val, max_val = sorted_population[0, dim], sorted_population[-1, dim]
            if max_val > min_val:
                distances[sorted_indices[1:-1]] += (sorted_population[2:, dim] - sorted_population[:-2, dim]) / (max_val - min_val)
        return distances

    def _constrain_to_periodic(self, population, bounds):
        """ Constrain solutions to encourage adaptive periodicity """
        lb, ub = bounds.lb, bounds.ub
        period = 2 + int((self.eval_count / self.budget) * (self.dim // 2))
        for i in range(0, population.shape[1], period):
            avg = np.mean(population[:, i:i+period], axis=1)
            population[:, i:i+period] = np.clip(np.repeat(avg[:, None], period, axis=1), lb[i:i+period], ub[i:i+period])
        return population

    def _differential_evolution(self, func, bounds, pop_size=20, max_iter=100):
        """ Differential Evolution with crowding and periodic constraints """
        lb, ub = bounds.lb, bounds.ub
        population = np.random.rand(pop_size, self.dim) * (ub - lb) + lb
        population = self._constrain_to_periodic(population, bounds)
        best_idx = np.argmin([self._reflectivity_cost(ind, func) for ind in population])
        best = population[best_idx].copy()

        for gen in range(max_iter):
            F = 0.5 + (0.9 - 0.5) * np.sin(gen / (max_iter * 2) * np.pi)  # Adjusted dynamic mutation factor
            CR = 0.5 + (0.9 - 0.5) * gen / max_iter
            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, population[i])
                trial = self._constrain_to_periodic(trial[None, :], bounds)[0]

                cost_trial = self._reflectivity_cost(trial, func)
                cost_i = self._reflectivity_cost(population[i], func)
                if cost_trial < cost_i or (cost_trial == cost_i and self._crowding_distance(np.vstack([population, trial]))[-1] > self._crowding_distance(population)[i]):
                    population[i] = trial
                    if cost_trial < self._reflectivity_cost(best, func):
                        best = trial.copy()

        return best

    def __call__(self, func):
        bounds = func.bounds
        # Global search with Crowding Differential Evolution
        best_global = self._differential_evolution(func, bounds)

        # Local search using BFGS
        result = minimize(lambda x: self._reflectivity_cost(x, func), best_global, method='L-BFGS-B', bounds=[(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)])
        
        return result.x if result.success else best_global