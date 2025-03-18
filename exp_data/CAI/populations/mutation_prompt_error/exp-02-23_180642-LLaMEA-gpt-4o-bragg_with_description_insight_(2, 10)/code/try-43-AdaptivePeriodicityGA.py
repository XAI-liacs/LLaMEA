import numpy as np
from scipy.optimize import minimize

class AdaptivePeriodicityGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.base_population_size = 10 * dim  # Modified: Introduced base_population_size
        self.population_size = self.base_population_size
        self.mutation_rate = 0.1
        self.crossover_rate = 0.9
        self.elite_rate = 0.05
        self.population = None
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))

    def adaptive_mutation_rate(self):
        self.mutation_rate = 0.05 + 0.5 * (1 - self.evaluations / self.budget)

    def adaptive_crossover_rate(self):
        self.crossover_rate = 0.7 + 0.2 * (1 - self.evaluations / self.budget)

    def dynamic_population_resizing(self):  # New function: Dynamic population resizing
        self.population_size = self.base_population_size + int(5 * np.sin(np.pi * self.evaluations / self.budget))

    def genetic_algorithm_step(self, func, lb, ub):
        new_population = np.empty_like(self.population)
        fitness = np.array([func(ind) for ind in self.population])
        best_idx = np.argmin(fitness)
        best_solution = self.population[best_idx]

        num_elites = max(1, int(self.population_size * self.elite_rate))
        elites = self.population[np.argsort(fitness)[:num_elites]]

        for i in range(num_elites, self.population_size):
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

        new_population[:num_elites] = elites

        return new_population, best_solution

    def impose_periodicity(self, solution, period_length=2):
        period_length = self.dim // 10 if self.evaluations > self.budget // 2 else period_length
        for i in range(0, self.dim, period_length):
            solution[i:i+period_length] = np.mean(solution[i:i+period_length])
        return solution

    def enhanced_local_search(self, func, solution, lb, ub):  # Modified: Improved local search with restart
        best_local_solution = solution
        best_local_value = func(solution)
        bounds = [(lb[i], ub[i]) for i in range(self.dim)]
        for _ in range(3):  # Allow multiple starts
            result = minimize(func, solution, method='L-BFGS-B', bounds=bounds)
            self.evaluations += result.nfev
            if result.fun < best_local_value:
                best_local_value = result.fun
                best_local_solution = result.x
            solution = np.random.uniform(lb, ub, self.dim)  # Restart with a random position

        return best_local_solution

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)

        best_solution = None
        best_value = float('inf')

        while self.evaluations < self.budget:
            self.adaptive_mutation_rate()
            self.adaptive_crossover_rate()
            self.dynamic_population_resizing()  # Call dynamic population resizing
            self.population, current_best = self.genetic_algorithm_step(func, lb, ub)

            candidate = self.enhanced_local_search(func, current_best, lb, ub)  # Use enhanced local search
            f_candidate = func(candidate)

            if f_candidate < best_value:
                best_value = f_candidate
                best_solution = candidate

            if self.evaluations >= self.budget:
                break

        return best_solution