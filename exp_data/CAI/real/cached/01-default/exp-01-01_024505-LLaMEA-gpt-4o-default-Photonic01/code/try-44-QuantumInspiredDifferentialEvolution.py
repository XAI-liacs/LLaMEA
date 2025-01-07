import numpy as np

class QuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, min(50, budget // 10))
        self.population = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.generational_progress = 0
        self.diversity_threshold = 0.1

    def initialize_population(self, lb, ub):
        self.population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)

    def evaluate_population(self, func):
        fitness = np.array([func(ind) for ind in self.population])
        best_index = np.argmin(fitness)
        if fitness[best_index] < self.best_fitness:
            self.best_fitness = fitness[best_index]
            self.best_solution = self.population[best_index]
        return fitness

    def mutate(self, indices, lb, ub):
        idx_a, idx_b, idx_c = indices
        a, b, c = self.population[idx_a], self.population[idx_b], self.population[idx_c]
        mutant_vector = a + self.mutation_factor * (b - c)
        return np.clip(mutant_vector, lb, ub)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        offspring = np.where(crossover_mask, mutant, target)
        return offspring

    def apply_gaussian_perturbation(self, individual, lb, ub):
        if np.random.rand() < 0.3:
            perturbation = np.random.normal(0, 0.1, size=self.dim)
            individual += perturbation
        return np.clip(individual, lb, ub)

    def adapt_population_size(self, lb, ub):
        diversity = np.mean(np.std(self.population, axis=0))
        if diversity < self.diversity_threshold:
            new_individuals = lb + (ub - lb) * np.random.rand(5, self.dim)
            self.population = np.vstack((self.population, new_individuals))
            self.population_size = len(self.population)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            fitness = self.evaluate_population(func)
            evaluations += len(fitness)

            if evaluations >= self.budget:
                break

            new_population = []
            for idx in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                mutant_vector = self.mutate(indices, lb, ub)
                target_vector = self.population[idx]
                offspring = self.crossover(target_vector, mutant_vector)
                offspring = self.apply_gaussian_perturbation(offspring, lb, ub)
                new_population.append(offspring)

            self.population = np.array(new_population)
            self.adapt_population_size(lb, ub)
            self.generational_progress = evaluations

        return self.best_solution, self.best_fitness