import numpy as np
from scipy.optimize import minimize

class HybridPSONM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.inertia = 0.65  # Slightly increase inertia for better exploration-exploitation balance
        self.cognitive_coef = 1.5  # Change line: Increase cognitive coefficient for enhanced exploration
        self.social_coef = 1.2
        self.init_budget = budget // 2
        self.local_budget = budget - self.init_budget
        self.velocities = np.zeros((self.population_size, self.dim))

    def _initialize_population(self, bounds):
        lb, ub = bounds
        return np.random.uniform(lb, ub, (self.population_size, self.dim)) * 0.8

    def _update_velocities(self, population, personal_best, global_best):
        r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
        self.velocities = (self.inertia * self.velocities + 
                           self.cognitive_coef * r1 * (personal_best - population) +
                           self.social_coef * r2 * (global_best - population))

    def _pso_step(self, population, func, bounds):
        lb, ub = bounds
        personal_best = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_scores)]

        new_population = np.copy(population)
        for _ in range(self.init_budget // self.population_size):
            self._update_velocities(population, personal_best, global_best)
            new_population = population + self.velocities
            new_population = np.clip(new_population, lb, ub)
            scores = np.array([func(ind) for ind in new_population])
            better_mask = scores < personal_best_scores
            personal_best_scores[better_mask] = scores[better_mask]
            personal_best[better_mask] = new_population[better_mask]
            if np.min(scores) < func(global_best):
                global_best = new_population[np.argmin(scores)]
            population = new_population
        return global_best

    def _encourage_periodicity(self, solution, bounds):
        lb, ub = bounds
        period = max(2, self.dim // 4)
        periodic_solution = np.tile(solution[:period], self.dim // period + 1)[:self.dim]
        return np.clip(periodic_solution, lb, ub)

    def _local_refinement(self, solution, func, bounds):
        result = minimize(func, solution, method='Nelder-Mead', options={'maxfev': self.local_budget})
        return result.x

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        population = self._initialize_population(bounds)

        best_solution = self._pso_step(population, func, bounds)
        best_solution = self._encourage_periodicity(best_solution, bounds)

        if self.local_budget > 0:
            best_solution = self._local_refinement(best_solution, func, bounds)

        return best_solution