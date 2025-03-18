import numpy as np
from scipy.optimize import minimize

class AdaptivePeriodicityGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_rate = 0.1
        self.crossover_rate = 0.9
        self.population = None
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))

    def adaptive_mutation_rate(self):
        # Adjust mutation rate based on progress
        self.mutation_rate = 0.1 + 0.4 * (1 - self.evaluations / self.budget)

    def adaptive_crossover_rate(self):
        # New dynamic crossover rate adjustment
        self.crossover_rate = 0.9 - 0.4 * (self.evaluations / self.budget)

    def genetic_algorithm_step(self, func, lb, ub):
        self.adaptive_crossover_rate()  # Ensure crossover rate is updated
        new_population = np.empty_like(self.population)
        fitness = np.array([func(ind) for ind in self.population])
        best_idx = np.argmin(fitness)
        best_solution = self.population[best_idx]

        for i in range(self.population_size):
            parents = np.random.choice(self.population_size, 2, replace=False)
            parent1, parent2 = self.population[parents]
            crossover_mask = np.random.rand(self.dim) < self.crossover_rate
            offspring = np.where(crossover_mask, parent1, parent2)
            mutation_mask = np.random.rand(self.dim) < self.mutation_rate
            offspring = np.where(mutation_mask, np.random.uniform(lb, ub), offspring)
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
        period_length = self.dim // 10 if self.evaluations > self.budget // 2 else period_length
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
            self.adaptive_mutation_rate()
            self.population, current_best = self.genetic_algorithm_step(func, lb, ub)

            candidate = self.local_search(func, current_best, lb, ub)
            f_candidate = func(candidate)

            if f_candidate < best_value:
                best_value = f_candidate
                best_solution = candidate

            if self.evaluations >= self.budget:
                break

        return best_solution