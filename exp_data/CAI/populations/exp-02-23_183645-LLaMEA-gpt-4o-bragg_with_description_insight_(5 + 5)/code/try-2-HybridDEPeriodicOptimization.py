import numpy as np
from scipy.optimize import minimize

class HybridDEPeriodicOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7

    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def evolve_population(self, population, func):
        new_population = np.copy(population)
        for i in range(self.population_size):
            indices = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
            x1, x2, x3 = population[indices]
            mutant_vector = x1 + self.mutation_factor * (x2 - x3)
            mutant_vector = np.clip(mutant_vector, func.bounds.lb, func.bounds.ub)
            
            cross_points = np.random.rand(self.dim) < self.crossover_rate
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
                
            trial_vector = np.where(cross_points, mutant_vector, population[i])
            if func(trial_vector) < func(population[i]):
                new_population[i] = trial_vector

        return new_population

    def periodic_local_search(self, individual, func):
        def periodicity_objective(x):
            regularity_penalty = np.sum((x[:-1] - x[1:])**2)
            return func(x) + 0.1 * regularity_penalty

        result = minimize(periodicity_objective, individual, bounds=[(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)], method='L-BFGS-B')
        return result.x

    def __call__(self, func):
        population = self.initialize_population(func.bounds.lb, func.bounds.ub)
        evaluations = 0
        best_solution = None
        best_value = float('inf')

        while evaluations < self.budget:
            population = self.evolve_population(population, func)
            evaluations += self.population_size

            for individual in population:
                optimized_individual = self.periodic_local_search(individual, func)
                individual_value = func(optimized_individual)
                evaluations += 1

                if individual_value < best_value:
                    best_value = individual_value
                    best_solution = optimized_individual

                if evaluations >= self.budget:
                    break

        return best_solution