import numpy as np

class HQGA:
    def __init__(self, budget, dim, population_size=20, mutation_rate=0.1, crossover_rate=0.7, quantum_rate=0.3):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.quantum_rate = quantum_rate
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_global_position = None
        best_global_value = float('inf')

        population = self.quantum_initialize_population(lb, ub)
        
        while self.evaluations < self.budget:
            fitness_values = np.array([func(individual) for individual in population])
            self.evaluations += self.population_size
            
            if np.min(fitness_values) < best_global_value:
                best_global_value = np.min(fitness_values)
                best_global_position = population[np.argmin(fitness_values)].copy()

            if self.evaluations >= self.budget:
                break

            selected = self.selection(population, fitness_values)
            offspring = self.crossover(selected)
            mutated_offspring = self.mutation(offspring, lb, ub)
            
            population = mutated_offspring

        return best_global_position

    def quantum_initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def selection(self, population, fitness_values):
        probabilities = 1 / (fitness_values + 1e-9)
        probabilities /= probabilities.sum()
        indices = np.random.choice(range(self.population_size), size=self.population_size, p=probabilities)
        return population[indices]

    def crossover(self, population):
        new_population = np.copy(population)
        for i in range(0, self.population_size - 1, 2):
            if np.random.rand() < self.crossover_rate:
                crossover_point = np.random.randint(1, self.dim)
                new_population[i, :crossover_point], new_population[i+1, :crossover_point] = (
                    new_population[i+1, :crossover_point], new_population[i, :crossover_point])
        return new_population

    def mutation(self, population, lb, ub):
        for i in range(self.population_size):
            if np.random.rand() < self.mutation_rate:
                mutation_vector = (np.random.rand(self.dim) - 0.5) * 0.1 * (ub - lb)
                population[i] = np.clip(population[i] + mutation_vector, lb, ub)
        return population