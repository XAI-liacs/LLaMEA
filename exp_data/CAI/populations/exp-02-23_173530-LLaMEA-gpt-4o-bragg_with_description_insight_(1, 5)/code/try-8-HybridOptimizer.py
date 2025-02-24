import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
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

    def _quasi_oppositional(self, population, bounds):
        """ Quasi-oppositional initialization """
        lb, ub = bounds.lb, bounds.ub
        pop_size = population.shape[0]
        opp_population = lb + ub - population + np.random.rand(pop_size, self.dim) * (ub - lb) * 0.1
        return np.clip(opp_population, lb, ub)

    def _differential_evolution(self, func, bounds, pop_size=20, max_iter=100):
        """ Differential Evolution with dynamic mutation and crossover, plus elitism """
        lb, ub = bounds.lb, bounds.ub
        population = np.random.rand(pop_size, self.dim) * (ub - lb) + lb
        population = np.concatenate((population, self._quasi_oppositional(population, bounds)))
        best_idx = np.argmin([self._reflectivity_cost(ind, func) for ind in population])
        best = population[best_idx].copy()
        
        for gen in range(max_iter):
            F = 0.6 + (0.8 - 0.6) * (max_iter - gen) / max_iter  # Adjusted mutation factor
            CR = 0.4 + (0.9 - 0.4) * gen / max_iter  # Adjusted crossover probability
            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, population[i])
                
                if self._reflectivity_cost(trial, func) < self._reflectivity_cost(population[i], func):
                    population[i] = trial
                    if self._reflectivity_cost(trial, func) < self._reflectivity_cost(best, func):
                        best = trial.copy()
            best = np.clip(best, lb, ub)  # Ensure best is within bounds

        return best

    def __call__(self, func):
        bounds = func.bounds
        # Global search with Differential Evolution
        best_global = self._differential_evolution(func, bounds)

        # Local search using BFGS
        result = minimize(lambda x: self._reflectivity_cost(x, func), best_global, method='L-BFGS-B', bounds=[(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)])
        
        return result.x if result.success else best_global