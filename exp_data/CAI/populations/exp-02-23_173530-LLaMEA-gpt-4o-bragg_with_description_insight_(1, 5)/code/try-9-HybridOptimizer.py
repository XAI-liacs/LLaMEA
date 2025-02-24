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
        """ Differential Evolution with dynamic mutation and crossover """
        lb, ub = bounds.lb, bounds.ub
        population = np.random.rand(pop_size, self.dim) * (ub - lb) + lb
        population = np.concatenate((population, self._quasi_oppositional(population, bounds)))
        scores = np.array([self._reflectivity_cost(ind, func) for ind in population])
        best_idx = np.argmin(scores)
        best = population[best_idx].copy()
        
        for gen in range(max_iter):
            F = 0.5 + (0.9 - 0.5) * (max_iter - gen) / max_iter  # Dynamic mutation factor
            CR = 0.5 + (0.9 - 0.5) * gen / max_iter  # Dynamic crossover probability
            diversity = np.std(scores)  # Calculate diversity
            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                if diversity < 1e-5:  # Adaptive mutation strategy
                    F *= 1.2
                mutant = np.clip(a + F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, population[i])
                
                trial_cost = self._reflectivity_cost(trial, func)
                if trial_cost < scores[i]:
                    population[i] = trial
                    scores[i] = trial_cost
                    if trial_cost < self._reflectivity_cost(best, func):
                        best = trial.copy()

        return best

    def __call__(self, func):
        bounds = func.bounds
        # Global search with Differential Evolution
        best_global = self._differential_evolution(func, bounds)

        # Local search using BFGS
        result = minimize(lambda x: self._reflectivity_cost(x, func), best_global, method='L-BFGS-B', bounds=[(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)])
        
        return result.x if result.success else best_global