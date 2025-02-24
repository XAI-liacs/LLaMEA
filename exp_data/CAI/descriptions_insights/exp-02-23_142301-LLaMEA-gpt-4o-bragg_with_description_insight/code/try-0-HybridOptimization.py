import numpy as np
from scipy.optimize import minimize

class HybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.scale_f = 0.8
        self.crossover_prob = 0.9
        self.used_evaluations = 0

    def _initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        return population

    def _evaluate_population(self, func, population):
        fitness = np.zeros(self.population_size)
        for i in range(self.population_size):
            if self.used_evaluations < self.budget:
                fitness[i] = func(population[i])
                self.used_evaluations += 1
        return fitness

    def _select_best(self, population, fitness):
        min_idx = np.argmin(fitness)
        return population[min_idx], fitness[min_idx]

    def _differential_evolution_step(self, population, fitness, bounds):
        lb, ub = bounds.lb, bounds.ub
        new_population = np.copy(population)
        for i in range(self.population_size):
            if self.used_evaluations >= self.budget:
                break
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + self.scale_f * (b - c), lb, ub)
            cross_points = np.random.rand(self.dim) < self.crossover_prob
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            if self.used_evaluations < self.budget:
                trial_fitness = func(trial)
                self.used_evaluations += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
        return new_population, fitness

    def _local_fine_tuning(self, best_solution, func, bounds):
        if self.used_evaluations >= self.budget:
            return best_solution
        result = minimize(func, best_solution, bounds=list(zip(bounds.lb, bounds.ub)), 
                          method='L-BFGS-B', options={'maxfun': self.budget - self.used_evaluations})
        self.used_evaluations += result.nfev
        return result.x if result.success else best_solution

    def __call__(self, func):
        bounds = func.bounds
        population = self._initialize_population(bounds)
        fitness = self._evaluate_population(func, population)
        best_solution, best_fitness = self._select_best(population, fitness)

        while self.used_evaluations < self.budget:
            population, fitness = self._differential_evolution_step(population, fitness, bounds)
            current_best, current_fitness = self._select_best(population, fitness)
            if current_fitness < best_fitness:
                best_solution, best_fitness = current_best, current_fitness

        best_solution = self._local_fine_tuning(best_solution, func, bounds)

        return best_solution