import numpy as np
from scipy.optimize import minimize

class HybridDEBFGS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_factor = 0.9  # Increased mutation factor for broader exploration
        self.crossover_prob = 0.7
        self.init_budget = budget // 2
        self.local_budget = budget - self.init_budget

    def _initialize_population(self, bounds):
        lb, ub = bounds
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def _mutate(self, target_idx, population, bounds):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = population[a] + self.mutation_factor * (population[b] - population[c])
        mutant = np.clip(mutant, bounds[0], bounds[1])
        return mutant

    def _crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_prob
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _de_step(self, population, func, bounds):
        new_population = np.zeros_like(population)
        for i in range(self.population_size):
            mutant = self._mutate(i, population, bounds)
            trial = self._crossover(population[i], mutant)
            if func(trial) < func(population[i]):
                new_population[i] = trial
            else:
                new_population[i] = population[i]
        return new_population

    def _encourage_periodicity(self, solution, bounds):
        lb, ub = bounds
        period = self.dim // 2
        periodic_solution = np.tile(solution[:period], self.dim // period + 1)[:self.dim]
        return np.clip(periodic_solution, lb, ub)

    def _local_refinement(self, solution, func, bounds):
        result = minimize(func, solution, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.local_budget})
        return result.x

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        population = self._initialize_population(bounds)
        evaluations = 0

        while evaluations < self.init_budget:
            population = self._de_step(population, func, bounds)
            evaluations += self.population_size

        best_idx = np.argmin([func(ind) for ind in population])
        best_solution = population[best_idx]
        best_solution = self._encourage_periodicity(best_solution, bounds)

        if self.local_budget > 0:
            best_solution = self._local_refinement(best_solution, func, bounds)

        return best_solution