import numpy as np
from scipy.optimize import minimize

class HybridPeriodicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def _initialize_population(self, pop_size, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (pop_size, self.dim))

    def _differential_evolution(self, population, bounds, F=0.8, CR=0.9):
        new_population = np.copy(population)
        elite_idx = np.argmin([self.func(ind) for ind in population])  # Identify elite candidate
        for i in range(len(population)):
            if i == elite_idx:  # Retain elite in population
                continue
            idxs = [idx for idx in range(len(population)) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            F_adaptive = 0.6 + 0.4 * np.random.rand()  # Slightly adjusted F range
            CR_adaptive = 0.4 + 0.6 * np.random.rand()  # Slightly adjusted CR range
            mutant_vector = np.clip(a + F_adaptive * (b - c), bounds.lb, bounds.ub)
            crossover = np.random.rand(self.dim) < CR_adaptive
            trial_vector = np.where(crossover, mutant_vector, population[i])
            trial_vector = self._enforce_periodicity(trial_vector)
            if self.func(trial_vector) < self.func(population[i]):
                new_population[i] = trial_vector
        return new_population

    def _enforce_periodicity(self, vector):
        period = max(1, self.dim // 3)  # Adjusted period calculation
        for i in range(self.dim - period):
            vector[i] = 0.5 * (vector[i] + vector[i % period])  # Averaged enforcement
        return vector

    def _local_optimization(self, vector, bounds):
        if np.random.rand() < 0.4:  # Increased local search frequency
            bounds_ = list(zip(bounds.lb, bounds.ub))
            result = minimize(self.func, vector, method='L-BFGS-B', bounds=bounds_)
            return result.x if result.success else vector
        return vector

    def __call__(self, func):
        self.func = func
        bounds = func.bounds
        pop_size = 10
        population = self._initialize_population(pop_size, bounds)
        self.evals += pop_size

        while self.evals < self.budget:
            population = self._differential_evolution(population, bounds)
            for i in range(len(population)):
                if self.evals >= self.budget:
                    break
                population[i] = self._local_optimization(population[i], bounds)
                self.evals += 1

        best_index = np.argmin([self.func(ind) for ind in population])
        return population[best_index]