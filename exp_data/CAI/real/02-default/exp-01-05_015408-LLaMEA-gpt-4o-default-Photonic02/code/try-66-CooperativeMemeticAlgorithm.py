import numpy as np

class CooperativeMemeticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 15 * dim
        self.local_search_rate = 0.3
        self.global_search_rate = 0.7
        self.local_improvement_range = 0.1
        self.global_mutation_strength = 0.5

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        best_index = np.argmin(fitness)
        best_position = population[best_index]

        while evaluations < self.budget:
            new_population = np.empty_like(population)

            for i in range(self.population_size):
                if np.random.rand() < self.local_search_rate:
                    new_population[i] = self.local_improvement(population[i], lb, ub)
                else:
                    new_population[i] = self.global_search(population[i], lb, ub)

                new_fitness = func(new_population[i])
                evaluations += 1

                if new_fitness < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness

                if new_fitness < fitness[best_index]:
                    best_index = i
                    best_position = new_population[i]

                if evaluations >= self.budget:
                    break

        return best_position, fitness[best_index]

    def local_improvement(self, position, lb, ub):
        perturbation = np.random.uniform(-self.local_improvement_range, self.local_improvement_range, self.dim)
        new_position = position + perturbation * (ub - lb)
        new_position = np.clip(new_position, lb, ub)
        return new_position

    def global_search(self, position, lb, ub):
        mutation = np.random.normal(0, self.global_mutation_strength, self.dim)
        new_position = position + mutation * (ub - lb)
        new_position = np.clip(new_position, lb, ub)
        return new_position