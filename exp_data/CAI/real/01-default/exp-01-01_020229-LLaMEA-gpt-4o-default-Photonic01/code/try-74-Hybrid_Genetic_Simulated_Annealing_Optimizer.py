import numpy as np

class Hybrid_Genetic_Simulated_Annealing_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.crossover_rate = 0.8
        self.mutation_rate = 0.2
        self.temperature_decay = 0.95
        self.initial_temperature = 1.0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        # Initialize population
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        
        evaluations = self.population_size
        temperature = self.initial_temperature

        while evaluations < self.budget:
            # Selection
            selected_indices = np.random.choice(self.population_size, self.population_size, replace=True, p=self._softmax(-fitness))
            selected_population = population[selected_indices]

            # Crossover
            offspring = np.copy(selected_population)
            for i in range(0, self.population_size, 2):
                if np.random.rand() < self.crossover_rate:
                    crossover_point = np.random.randint(1, self.dim)
                    offspring[i, crossover_point:], offspring[i+1, crossover_point:] = \
                        selected_population[i+1, crossover_point:], selected_population[i, crossover_point:]

            # Mutation
            for i in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    mutation_vector = np.random.normal(scale=0.1, size=self.dim)
                    offspring[i] += mutation_vector
                    offspring[i] = np.clip(offspring[i], lb, ub)

            # Evaluate offspring
            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations += self.population_size

            # Simulated Annealing: Accept new solutions based on temperature
            for i in range(self.population_size):
                delta_fitness = offspring_fitness[i] - fitness[selected_indices[i]]
                if delta_fitness < 0 or np.random.rand() < np.exp(-delta_fitness / temperature):
                    population[selected_indices[i]] = offspring[i]
                    fitness[selected_indices[i]] = offspring_fitness[i]
                    if fitness[selected_indices[i]] < best_fitness:
                        best_solution = population[selected_indices[i]]
                        best_fitness = fitness[selected_indices[i]]

            # Update temperature
            temperature *= self.temperature_decay

        return best_solution, best_fitness

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)