import numpy as np
from scipy.optimize import minimize

class AdaptiveGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_rate = 0.05  # Reduced from 0.1 to 0.05
        self.crossover_rate = 0.75
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub

        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += self.population_size

        while self.evaluations < self.budget:
            selection_prob = fitness / np.sum(fitness)
            indices = np.random.choice(self.population_size, self.population_size, p=selection_prob)

            new_population = np.empty_like(population)
            for i in range(0, self.population_size, 2):
                parent1, parent2 = population[indices[i]], population[indices[i + 1]]
                if np.random.rand() < self.crossover_rate:
                    crossover_point = np.random.randint(1, self.dim)
                    new_population[i] = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                    new_population[i + 1] = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
                else:
                    new_population[i], new_population[i + 1] = parent1, parent2

            for individual in new_population:
                if np.random.rand() < self.mutation_rate:
                    mutation_index = np.random.randint(self.dim)
                    individual[mutation_index] = np.random.uniform(lb[mutation_index], ub[mutation_index])

            for i, individual in enumerate(new_population):
                if np.random.rand() < 0.8:
                    shift = np.random.randint(1, self.dim // 2)
                    period = individual[:shift]
                    individual[:shift], individual[-shift:] = period, period

            new_fitness = np.array([func(ind) for ind in new_population])
            self.evaluations += self.population_size

            combined_population = np.vstack((population, new_population))
            combined_fitness = np.hstack((fitness, new_fitness))
            best_indices = np.argsort(combined_fitness)[-self.population_size:]
            population, fitness = combined_population[best_indices], combined_fitness[best_indices]

            for i in range(self.population_size):
                if np.random.rand() < 0.15 and self.evaluations < self.budget:
                    res = minimize(func, population[i], bounds=list(zip(lb, ub)), method='L-BFGS-B')
                    if res.success:
                        population[i] = res.x
                        fitness[i] = res.fun
                        self.evaluations += res.nfev

            # Apply elitism to retain the best solution found so far
            best_index = np.argmax(fitness)
            elite_individual = population[best_index]
            elite_fitness = fitness[best_index]

        return elite_individual