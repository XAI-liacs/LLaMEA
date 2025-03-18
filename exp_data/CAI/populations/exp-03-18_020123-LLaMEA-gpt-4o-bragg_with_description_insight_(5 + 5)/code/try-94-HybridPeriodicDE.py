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

    def _adaptive_crossover_rate(self, iter_count):
        # Adaptive crossover rate that increases with iterations
        return 0.5 + 0.4 * (iter_count / self.budget)

    def _differential_evolution(self, population, bounds, F=0.8):
        new_population = np.copy(population)
        for i in range(len(population)):
            idxs = [idx for idx in range(len(population)) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant_vector = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
            CR = self._adaptive_crossover_rate(self.evals)  # Changed line
            crossover = np.random.rand(self.dim) < CR  # Changed line
            trial_vector = np.where(crossover, mutant_vector, population[i])
            trial_vector = self._enforce_periodicity(trial_vector, self.evals, population) 
            if self.func(trial_vector) < self.func(population[i]):
                new_population[i] = trial_vector
        return new_population

    def _enforce_periodicity(self, vector, iter_count, population): 
        diversity = np.std(population, axis=0).mean()
        period = max(1, self.dim // (iter_count + 1)) 
        scaling_factor = 0.005 * np.sin(iter_count / self.budget * np.pi)
        for i in range(self.dim - period):
            vector[i] = vector[i % period] + diversity * scaling_factor
        return vector

    def _local_optimization(self, vector, bounds):
        bounds_ = list(zip(bounds.lb, bounds.ub))
        result = minimize(self.func, vector, method='L-BFGS-B', bounds=bounds_)
        return result.x if result.success else vector

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