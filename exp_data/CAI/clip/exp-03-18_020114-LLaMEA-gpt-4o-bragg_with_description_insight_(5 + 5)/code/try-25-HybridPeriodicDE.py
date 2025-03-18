import numpy as np
from scipy.optimize import minimize

class HybridPeriodicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.f = 0.5  # Scaling factor for DE
        self.cr = 0.75  # Crossover rate for DE
        self.evaluations = 0

    def _initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        return pop

    def _mutate(self, pop, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = pop[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + self.f * (b - c), -1, 1)
        return mutant

    def _crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.cr
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _local_search(self, current, bounds, func):
        # Enforce periodicity by using a cost function
        def periodic_cost(x):
            periodicity_penalty = np.sum((x - np.roll(x, 1))**2)
            return func(x) + 0.1 * periodicity_penalty

        result = minimize(periodic_cost, current, bounds=bounds, method='L-BFGS-B')
        return result.x

    def __call__(self, func):
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        population = self._initialize_population(func.bounds)
        best_solution = None
        best_value = float('inf')

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                target = population[i]
                mutant = self._mutate(population, i)
                trial = self._crossover(target, mutant)

                # Ensure trial is within bounds
                trial = np.clip(trial, func.bounds.lb, func.bounds.ub)
                trial_value = func(trial)
                self.evaluations += 1

                if trial_value < func(target):
                    population[i] = trial

                # Perform local search at intervals
                if self.evaluations % (self.population_size // 2) == 0 and self.evaluations < self.budget:
                    population[i] = self._local_search(population[i], bounds, func)
                    self.evaluations += 1

                trial_value = func(population[i])
                if trial_value < best_value:
                    best_value = trial_value
                    best_solution = population[i]

        return best_solution