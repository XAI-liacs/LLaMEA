import numpy as np
from scipy.optimize import minimize

class AdaptiveDEWithPeriodicity:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = max(10, 5 * dim)
        self.scale_f = 0.5
        self.crossover_prob = 0.7
        self.used_evaluations = 0

    def _adaptive_population_size(self):
        return min(self.initial_population_size + self.used_evaluations // (5 * self.dim), 50 * self.dim)
    
    def _initialize_population(self, bounds, pop_size):
        lb, ub = bounds.lb, bounds.ub
        population = lb + (ub - lb) * np.random.rand(pop_size, self.dim)
        return population

    def _evaluate_population(self, func, population):
        fitness = np.zeros(len(population))
        for i in range(len(population)):
            if self.used_evaluations < self.budget:
                fitness[i] = func(population[i])
                self.used_evaluations += 1
        return fitness

    def _select_best(self, population, fitness):
        min_idx = np.argmin(fitness)
        return population[min_idx], fitness[min_idx]

    def _differential_evolution_step(self, func, population, fitness, bounds):
        lb, ub = bounds.lb, bounds.ub
        new_population = np.copy(population)
        dynamic_periodicity_factor = 1 + np.sin(self.used_evaluations * np.pi / (self.budget / 1.5))
        for i in range(len(population)):
            if self.used_evaluations >= self.budget:
                break
            idxs = [idx for idx in range(len(population)) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            self.scale_f = 0.4 + 0.25 * np.cos(self.used_evaluations * np.pi / dynamic_periodicity_factor)  # Adjust scale factor
            mutant = np.clip(a + self.scale_f * (b - c), lb, ub)
            self.crossover_prob = 0.5 + 0.45 * np.cos(self.used_evaluations * np.pi / (self.budget / 1.5))  # More aggressive crossover strategy
            cross_points = np.random.rand(self.dim) < self.crossover_prob
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            elite_influence = 0.1 + 0.3 * np.sin(self.used_evaluations * np.pi / self.budget) * dynamic_periodicity_factor  # Dynamic elite influence
            trial = trial + (np.mean(population) - trial) * 0.15 * np.cos(self.used_evaluations * np.pi / self.budget)  # Encourage periodicity
            trial = np.clip(trial + elite_influence * (self._select_best(population, fitness)[0] - trial), lb, ub)
            if self.used_evaluations < self.budget:
                trial_fitness = func(trial)
                self.used_evaluations += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
        elite_idx = np.argmin(fitness)
        new_population[0] = population[elite_idx]
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
        population_size = self._adaptive_population_size()
        population = self._initialize_population(bounds, population_size)
        fitness = self._evaluate_population(func, population)
        best_solution, best_fitness = self._select_best(population, fitness)

        while self.used_evaluations < self.budget:
            population_size = self._adaptive_population_size()
            population, fitness = self._differential_evolution_step(func, population, fitness, bounds)
            current_best, current_fitness = self._select_best(population, fitness)
            if current_fitness < best_fitness:
                best_solution, best_fitness = current_best, current_fitness

        best_solution = self._local_fine_tuning(best_solution, func, bounds)

        return best_solution