import numpy as np
from scipy.optimize import minimize

class CoevolutionaryDualGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.main_population_size = 5 * dim
        self.aux_population_size = 5 * dim
        self.mutation_rate_main = 0.1
        self.crossover_rate_main = 0.9
        self.mutation_rate_aux = 0.2
        self.crossover_rate_aux = 0.85
        self.main_population = None
        self.aux_population = None
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        self.main_population = np.random.uniform(lb, ub, (self.main_population_size, self.dim))
        self.aux_population = np.random.uniform(lb, ub, (self.aux_population_size, self.dim))

    def adaptive_mutation_rate(self):
        self.mutation_rate_main = 0.1 + 0.4 * (1 - self.evaluations / self.budget)
        self.mutation_rate_aux = 0.2 + 0.3 * (1 - self.evaluations / self.budget)

    def genetic_algorithm_step(self, func, lb, ub):
        new_main_population = np.empty_like(self.main_population)
        new_aux_population = np.empty_like(self.aux_population)

        fitness_main = np.array([func(ind) for ind in self.main_population])
        fitness_aux = np.array([func(ind) for ind in self.aux_population])

        best_main_idx = np.argmin(fitness_main)
        best_main_solution = self.main_population[best_main_idx]

        best_aux_idx = np.argmin(fitness_aux)
        best_aux_solution = self.aux_population[best_aux_idx]

        # Evolve main population
        for i in range(self.main_population_size):
            parents = np.random.choice(self.main_population_size, 2, replace=False)
            parent1, parent2 = self.main_population[parents]
            crossover_mask = np.random.rand(self.dim) < self.crossover_rate_main
            offspring = np.where(crossover_mask, parent1, parent2)
            mutation_mask = np.random.rand(self.dim) < self.mutation_rate_main
            offspring = np.where(mutation_mask, np.random.uniform(lb, ub), offspring)
            offspring = self.impose_periodicity(offspring)

            if func(offspring) < fitness_main[i]:
                new_main_population[i] = offspring
            else:
                new_main_population[i] = self.main_population[i]

            self.evaluations += 1
            if self.evaluations >= self.budget:
                return new_main_population, best_main_solution

        # Evolve auxiliary population
        for i in range(self.aux_population_size):
            parents = np.random.choice(self.aux_population_size, 2, replace=False)
            parent1, parent2 = self.aux_population[parents]
            crossover_mask = np.random.rand(self.dim) < self.crossover_rate_aux
            offspring = np.where(crossover_mask, parent1, parent2)
            mutation_mask = np.random.rand(self.dim) < self.mutation_rate_aux
            offspring = np.where(mutation_mask, np.random.uniform(lb, ub), offspring)
            offspring = self.impose_periodicity(offspring)

            if func(offspring) < fitness_aux[i]:
                new_aux_population[i] = offspring
            else:
                new_aux_population[i] = self.aux_population[i]

            self.evaluations += 1
            if self.evaluations >= self.budget:
                return new_aux_population, best_aux_solution

        return new_main_population, new_aux_population, best_main_solution, best_aux_solution

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
            self.main_population, self.aux_population, main_best, aux_best = self.genetic_algorithm_step(func, lb, ub)

            # Local search on best solutions from both populations
            candidate_main = self.local_search(func, main_best, lb, ub)
            candidate_aux = self.local_search(func, aux_best, lb, ub)

            f_candidate_main = func(candidate_main)
            f_candidate_aux = func(candidate_aux)

            if f_candidate_main < best_value:
                best_value = f_candidate_main
                best_solution = candidate_main

            if f_candidate_aux < best_value:
                best_value = f_candidate_aux
                best_solution = candidate_aux

            if self.evaluations >= self.budget:
                break

        return best_solution