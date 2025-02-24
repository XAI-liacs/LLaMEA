import numpy as np
from scipy.optimize import minimize

class HybridPhotonicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.strategy = 'best1bin'
        self.mutation_factor = 0.8
        self.crossover_prob = 0.7

    def _initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        center = (ub + lb) / 2
        range_half = (ub - lb) / 2
        population = np.random.uniform(low=lb, high=ub, size=(self.population_size, self.dim))
        quasi_opposite_population = center + range_half - (population - center)
        return np.vstack((population, quasi_opposite_population))

    def _adaptive_mutation(self, fitness, iteration):
        return self.mutation_factor * (1 - iteration / self.budget)  # Decay mutation factor over time

    def _differential_evolution(self, func, bounds):
        population = self._initialize_population(bounds)
        fitness = np.array([func(ind) for ind in population])
        for iteration in range(int(self.budget / len(population))):
            for i, target in enumerate(population):
                candidates = list(range(len(population)))
                candidates.remove(i)
                a, b, c = population[np.random.choice(candidates, 3, replace=False)]
                mutation_factor = self._adaptive_mutation(fitness, iteration)
                mutant = np.clip(a + mutation_factor * (b - c), bounds.lb, bounds.ub)
                cross_points = np.random.rand(self.dim) < self.crossover_prob
                trial = np.where(cross_points, mutant, target)
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
        return population[np.argmin(fitness)], np.min(fitness)

    def _local_search(self, func, candidate, bounds):
        period = 2
        periodic_candidate = np.copy(candidate)
        for i in range(0, self.dim, period):
            periodic_candidate[i:i + period] = np.mean(candidate[i:i + period])
        result = minimize(func, periodic_candidate, bounds=np.array([bounds.lb, bounds.ub]).T, method='L-BFGS-B')
        return result.x, result.fun

    def __call__(self, func):
        best_sol, best_fitness = self._differential_evolution(func, func.bounds)
        local_sol, local_fitness = self._local_search(func, best_sol, func.bounds)
        if local_fitness < best_fitness:
            return local_sol, local_fitness
        else:
            return best_sol, best_fitness