import numpy as np
from scipy.optimize import minimize

class EnhancedCrowdingDEOptimizer:
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
        return distances * (1 + 0.15 * (self.eval_count / self.budget))

    def _constrain_to_periodic(self, population, bounds):
        """ Constrain solutions to encourage adaptive periodicity """
        lb, ub = bounds.lb, bounds.ub
        period = 2 + int((self.eval_count / self.budget) * (self.dim // 3))  # Adjusted periodicity calculation
        for i in range(0, self.dim, period):
            end = min(i + period, self.dim)
            avg = np.mean(population[:, i:end], axis=1)
            population[:, i:end] = np.clip(np.stack([avg] * (end - i), axis=1), lb[i:end], ub[i:end])
        return population

    def _differential_evolution(self, func, bounds, pop_size=20, max_iter=100):
        """ Differential Evolution with crowding and periodic constraints """
        lb, ub = bounds.lb, bounds.ub
        population = np.random.rand(pop_size, self.dim) * (ub - lb) + lb
        population = self._constrain_to_periodic(population, bounds)
        costs = np.array([self._reflectivity_cost(ind, func) for ind in population])
        best_idx = np.argmin(costs)
        best = population[best_idx].copy()

        for gen in range(max_iter):
            F = 0.5 + (0.9 - 0.5) * np.sin(gen / max_iter * np.pi/2)  # Modified F calculation
            CR = 0.5 + (0.9 - 0.5) * np.cos(gen / max_iter * np.pi/2)  # Minor adjustment in CR calculation
            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, population[i])
                trial = self._constrain_to_periodic(trial[None, :], bounds)[0]

                cost_trial = self._reflectivity_cost(trial, func)
                if cost_trial < costs[i] or (cost_trial == costs[i] and self._crowding_distance(np.vstack([population, trial]))[-1] > self._crowding_distance(population)[i]):
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