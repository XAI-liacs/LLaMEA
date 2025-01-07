import numpy as np

class Adaptive_Quantum_Inspired_Evolutionary_Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.alpha = 0.5  # balance between exploration and exploitation
        self.beta = 2.0   # scaling factor for quantum superposition
        self.mutation_rate = 0.1
        self.adaptive_rate = 0.95

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        # Initial population
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(indiv) for indiv in population])
        evaluations = self.population_size

        best_individual = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)

        while evaluations < self.budget:
            # Quantum-inspired crossover
            new_population = []
            for _ in range(self.population_size):
                parents = np.random.choice(range(self.population_size), size=2, replace=False)
                parent1, parent2 = population[parents[0]], population[parents[1]]
                gamma = np.random.rand(self.dim)
                offspring = (gamma * parent1 + (1 - gamma) * parent2) + \
                            self.beta * np.random.normal(0, 1, self.dim)
                offspring = np.clip(offspring, lb, ub)
                new_population.append(offspring)

            # Mutation
            for i in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    mutation_vector = np.random.normal(0, 0.1, self.dim)
                    new_population[i] += mutation_vector
                    new_population[i] = np.clip(new_population[i], lb, ub)

            # Combine and select
            combined_population = np.vstack((population, np.array(new_population)))
            combined_fitness = np.array([func(indiv) for indiv in combined_population])
            evaluations += len(combined_population)
            selected_indices = np.argsort(combined_fitness)[:self.population_size]
            population = combined_population[selected_indices]
            fitness = combined_fitness[selected_indices]

            current_best = np.min(fitness)
            if current_best < best_fitness:
                best_fitness = current_best
                best_individual = population[np.argmin(fitness)]

            # Dynamic adjustment of parameters
            self.beta *= self.adaptive_rate
            self.mutation_rate *= self.adaptive_rate

            if evaluations >= self.budget:
                break

        return best_individual, best_fitness