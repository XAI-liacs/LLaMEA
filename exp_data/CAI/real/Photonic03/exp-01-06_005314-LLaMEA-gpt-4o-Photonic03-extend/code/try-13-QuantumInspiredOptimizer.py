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
            neighborhood_size = (ub - lb) * (0.05 + 0.95 * (1 - self.evaluations / self.budget))
            population_size = int(initial_population_size * (1 + 0.5 * (1 - self.evaluations / self.budget)))
            if population_size < len(population):
                population = population[:population_size]
                fitness = fitness[:population_size]
            else:
                new_individuals = np.random.uniform(lb, ub, (population_size - len(population), self.dim))
                new_fitness = np.array([func(ind) for ind in new_individuals])
                self.evaluations += len(new_individuals)
                population = np.vstack((population, new_individuals))
                fitness = np.concatenate((fitness, new_fitness))
            
            # Quantum-inspired superposition and adaptive mutation
            best_fitness = np.min(fitness)
            diversity_factor = np.std(fitness) / (np.mean(fitness) + 1e-10)  # new line
            for i in range(population_size):
                if self.evaluations >= self.budget:
                    break
                adaptive_mutation = 0.05 + 0.95 * (fitness[i] - best_fitness) / (np.max(fitness) - best_fitness + 1e-10)
                adaptive_mutation *= (1 + diversity_factor)  # changed line
                new_candidate = population[i] + np.random.uniform(-adaptive_mutation, adaptive_mutation, self.dim) * neighborhood_size
                new_candidate = np.clip(new_candidate, lb, ub)
                new_fitness = func(new_candidate)
                self.evaluations += 1
                
                if new_fitness < fitness[i]:
                    population[i] = new_candidate
                    fitness[i] = new_fitness

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]