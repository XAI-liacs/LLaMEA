import numpy as np
from scipy.optimize import minimize

class AdaptiveGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_rate = 0.1
        self.crossover_rate = 0.75 
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub

        # Dual initialization with symmetric population
        population = np.random.uniform(lb, ub, (self.population_size//2, self.dim))
        population = np.vstack([population, lb + ub - population])  # Symmetrical counterpart
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += self.population_size

        while self.evaluations < self.budget:
            total_fitness = np.sum(fitness)
            selection_prob = fitness / total_fitness
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

            # Refined mutation strategy
            for individual in new_population:
                if np.random.rand() < self.mutation_rate:
                    mutation_indices = np.random.choice(self.dim, 2, replace=False)
                    individual[mutation_indices] = np.random.uniform(lb[mutation_indices], ub[mutation_indices])

            for individual in new_population:
                if np.random.rand() < 0.8:
                    shift = np.random.randint(1, self.dim // 2)
                    individual[:shift] = individual[-shift:]

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

        best_index = np.argmax(fitness)
        return population[best_index]