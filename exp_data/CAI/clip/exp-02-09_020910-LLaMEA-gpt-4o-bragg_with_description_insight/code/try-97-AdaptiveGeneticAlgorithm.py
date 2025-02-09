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

        # Initialize population with enhanced diversity
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        if self.population_size > 2:
            population[:2] = np.array([(lb + ub) / 2, np.random.uniform(lb, ub)])

        fitness = np.array([func(ind) for ind in population])
        self.evaluations += self.population_size

        while self.evaluations < self.budget:
            # Parent selection (roulette wheel)
            total_fitness = np.sum(fitness)
            selection_prob = fitness / total_fitness
            indices = np.random.choice(self.population_size, self.population_size, p=selection_prob)

            # Apply crossover with adaptive rate
            adaptive_crossover = np.clip(self.crossover_rate * (1 - (self.evaluations / self.budget)), 0.6, 0.9)
            new_population = np.empty_like(population)
            for i in range(0, self.population_size, 2):
                parent1, parent2 = population[indices[i]], population[indices[i + 1]]
                if np.random.rand() < adaptive_crossover:
                    crossover_point = np.random.randint(1, self.dim)
                    new_population[i] = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                    new_population[i + 1] = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
                else:
                    new_population[i], new_population[i + 1] = parent1, parent2

            # Apply adaptive mutation
            adaptive_mutation = np.clip(self.mutation_rate * (1 - (self.evaluations / self.budget)), 0.05, 0.2)
            for individual in new_population:
                if np.random.rand() < adaptive_mutation:
                    mutation_index = np.random.randint(self.dim)
                    individual[mutation_index] = np.random.uniform(lb[mutation_index], ub[mutation_index])

            # Enhanced Periodicity preservation
            for individual in new_population:
                if np.random.rand() < 0.8:
                    shift = np.random.randint(1, self.dim // 2)
                    individual[:shift] = individual[-shift:]

            # Evaluate new population
            new_fitness = np.array([func(ind) for ind in new_population])
            self.evaluations += self.population_size

            # Selection
            combined_population = np.vstack((population, new_population))
            combined_fitness = np.hstack((fitness, new_fitness))
            best_indices = np.argsort(combined_fitness)[-self.population_size:]
            population, fitness = combined_population[best_indices], combined_fitness[best_indices]

            # Local refinement using BFGS with adaptive inertia
            for i in range(self.population_size):
                if np.random.rand() < 0.15 and self.evaluations < self.budget:
                    inertia_weight = 1.0 - (self.evaluations / self.budget)  # Adaptive inertia
                    res = minimize(func, population[i], bounds=list(zip(lb, ub)), method='L-BFGS-B', options={'disp': False, 'maxcor': 10 * int(inertia_weight)})
                    if res.success:
                        population[i] = res.x
                        fitness[i] = res.fun
                        self.evaluations += res.nfev

        best_index = np.argmax(fitness)
        return population[best_index]