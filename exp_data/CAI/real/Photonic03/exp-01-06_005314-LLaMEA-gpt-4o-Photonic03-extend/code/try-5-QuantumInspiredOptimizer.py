import numpy as np

class QuantumInspiredOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        initial_population_size = 10
        population = np.random.uniform(lb, ub, (initial_population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += initial_population_size

        while self.evaluations < self.budget:
            progress_ratio = self.evaluations / self.budget
            neighborhood_size = (ub - lb) * (0.1 + 0.8 * (1 - progress_ratio ** 2))
            population_size = int(initial_population_size * (1 + 0.5 * (1 - progress_ratio ** 1.5)))
            if population_size < len(population):
                population = population[:population_size]
                fitness = fitness[:population_size]
            else:
                new_individuals = np.random.uniform(lb, ub, (population_size - len(population), self.dim))
                new_fitness = np.array([func(ind) for ind in new_individuals])
                self.evaluations += len(new_individuals)
                population = np.vstack((population, new_individuals))
                fitness = np.concatenate((fitness, new_fitness))
            
            best_fitness = np.min(fitness)
            for i in range(population_size):
                if self.evaluations >= self.budget:
                    break
                adaptive_mutation = 0.1 + 0.9 * (fitness[i] - best_fitness) / (np.max(fitness) - best_fitness + 1e-10)
                distance_to_bounds = np.minimum(population[i] - lb, ub - population[i])
                mutation_strength = np.random.uniform(-adaptive_mutation, adaptive_mutation, self.dim) * neighborhood_size * distance_to_bounds / (ub - lb)
                new_candidate = population[i] + mutation_strength
                new_candidate = np.clip(new_candidate, lb, ub)
                new_fitness = func(new_candidate)
                self.evaluations += 1
                
                if new_fitness < fitness[i]:
                    population[i] = new_candidate
                    fitness[i] = new_fitness

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]