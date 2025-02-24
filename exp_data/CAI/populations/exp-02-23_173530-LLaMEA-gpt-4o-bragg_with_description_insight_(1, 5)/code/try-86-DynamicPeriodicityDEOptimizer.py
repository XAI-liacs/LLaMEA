import numpy as np
from scipy.optimize import minimize

class DynamicPeriodicityDEOptimizer:
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

    def _dynamic_periodicity_constrain(self, population, bounds):
        """ Encourage solutions to adopt adaptive periodicity based on progress """
        lb, ub = bounds.lb, bounds.ub
        period = 3 + int((self.eval_count / self.budget) * (self.dim // 2))
        for i in range(0, self.dim, period):
            end = min(i + period, self.dim)
            avg = np.mean(population[:, i:end], axis=1)
            population[:, i:end] = np.clip(np.stack([avg] * (end - i), axis=1), lb[i:end], ub[i:end])
        return population

    def _cosine_similarity(self, a, b):
        """ Compute cosine similarity between two vectors """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

    def _select_parents(self, population, costs):
        """ Select parents based on cosine similarity to the best solution """
        best_idx = np.argmin(costs)
        best = population[best_idx]
        similarities = np.array([self._cosine_similarity(best, ind) for ind in population])
        probs = similarities / np.sum(similarities)
        indices = np.random.choice(len(population), size=3, p=probs, replace=False)
        return population[indices]

    def _differential_evolution(self, func, bounds, pop_size=20, max_iter=100):
        """ Differential Evolution with dynamic periodicity and cosine similarity selection """
        lb, ub = bounds.lb, bounds.ub
        population = np.random.rand(pop_size, self.dim) * (ub - lb) + lb
        population = self._dynamic_periodicity_constrain(population, bounds)
        costs = np.array([self._reflectivity_cost(ind, func) for ind in population])
        best_idx = np.argmin(costs)
        best = population[best_idx].copy()

        for gen in range(max_iter):
            F = 0.5 + 0.5 * np.cos(gen / max_iter * np.pi)
            CR = 0.8
            for i in range(pop_size):
                a, b, c = self._select_parents(population, costs)
                mutant = np.clip(a + F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, population[i])
                trial = self._dynamic_periodicity_constrain(trial[None, :], bounds)[0]

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