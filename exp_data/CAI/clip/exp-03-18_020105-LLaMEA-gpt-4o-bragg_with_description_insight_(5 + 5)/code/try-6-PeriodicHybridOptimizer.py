import numpy as np
from scipy.optimize import minimize

class PeriodicHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.func_evals = 0

    def _initialize_population(self, lb, ub):
        return lb + (ub - lb) * np.random.rand(self.population_size, self.dim)

    def _mutate(self, population, best_idx):
        idxs = np.random.choice(np.arange(self.population_size), 3, replace=False)
        a, b, c = population[idxs[0]], population[idxs[1]], population[idxs[2]]
        mutant = a + self.mutation_factor * (b - c)
        return np.clip(mutant, lb, ub)

    def _crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _periodic_cost(self, solution, func):
        # Promote periodicity by penalizing deviations from periodic patterns
        period_length = self.dim // 2
        periodic_parts = [solution[i:i+period_length] for i in range(0, self.dim, period_length)]
        periodicity_cost = np.sum([np.linalg.norm(part - np.mean(periodic_parts, axis=0)) for part in periodic_parts])
        return periodicity_cost + func(solution)

    def _local_search(self, solution, func):
        res = minimize(lambda x: func(x), solution, method='L-BFGS-B', bounds=list(zip(lb, ub)))
        return res.x if res.success else solution

    def __call__(self, func):
        global lb, ub
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self._initialize_population(lb, ub)
        best_idx = np.argmin([func(ind) for ind in population])
        best_solution = population[best_idx]

        while self.func_evals < self.budget:
            for i in range(self.population_size):
                mutant = self._mutate(population, best_idx)
                trial = self._crossover(population[i], mutant)
                trial_cost = self._periodic_cost(trial, func)
                target_cost = func(population[i])

                if trial_cost < target_cost:
                    population[i] = trial
                    if trial_cost < func(best_solution):
                        best_solution = trial
                        best_solution = self._local_search(best_solution, func)
                
                self.func_evals += 1
                if self.func_evals >= self.budget:
                    break

        return best_solution