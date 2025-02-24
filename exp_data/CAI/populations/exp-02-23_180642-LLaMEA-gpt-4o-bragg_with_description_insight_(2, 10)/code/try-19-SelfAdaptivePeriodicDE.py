import numpy as np
from scipy.optimize import minimize

class SelfAdaptivePeriodicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.population = None
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))

    def self_adaptive_parameters(self):
        # Self-adapt mutation factor and crossover rate based on evaluations
        self.mutation_factor = 0.5 + 0.5 * np.random.rand()
        self.crossover_rate = 0.5 + 0.5 * np.random.rand()

    def differential_evolution_step(self, func, lb, ub):
        new_population = np.empty_like(self.population)
        fitness = np.array([func(ind) for ind in self.population])
        best_idx = np.argmin(fitness)
        best_solution = self.population[best_idx]

        for i in range(self.population_size):
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
            mutant = np.clip(mutant, lb, ub)

            crossover_mask = np.random.rand(self.dim) < self.crossover_rate
            offspring = np.where(crossover_mask, mutant, self.population[i])
            offspring = self.impose_periodicity(offspring)

            if func(offspring) < fitness[i]:
                new_population[i] = offspring
            else:
                new_population[i] = self.population[i]

            self.evaluations += 1
            if self.evaluations >= self.budget:
                return new_population, best_solution

        return new_population, best_solution

    def impose_periodicity(self, solution, period_length=2):
        for i in range(0, self.dim, period_length):
            solution[i:i+period_length] = np.mean(solution[i:i+period_length])
        return solution

    def local_search(self, func, solution, lb, ub):
        bounds = [(lb[i], ub[i]) for i in range(self.dim)]
        result = minimize(func, solution, method='L-BFGS-B', bounds=bounds)
        self.evaluations += result.nfev
        return result.x

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)

        best_solution = None
        best_value = float('inf')

        while self.evaluations < self.budget:
            self.self_adaptive_parameters()
            self.population, current_best = self.differential_evolution_step(func, lb, ub)

            candidate = self.local_search(func, current_best, lb, ub)
            f_candidate = func(candidate)

            if f_candidate < best_value:
                best_value = f_candidate
                best_solution = candidate

            if self.evaluations >= self.budget:
                break

        return best_solution