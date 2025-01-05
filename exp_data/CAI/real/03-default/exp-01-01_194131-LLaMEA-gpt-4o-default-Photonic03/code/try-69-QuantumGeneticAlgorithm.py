import numpy as np

class QuantumGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(20, 10 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.fitness = np.full(self.population_size, float('inf'))
        self.best_individual = None
        self.best_fitness = float('inf')
        self.mutation_rate = 0.05
        self.elite_fraction = 0.1
        self.evaluations = 0

    def quantum_bit_flip(self, individual):
        mutation_vector = np.random.rand(self.dim) < self.mutation_rate
        new_individual = np.where(mutation_vector, 1 - individual, individual)
        return new_individual

    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, self.dim - 1)
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        return child

    def select_parents(self):
        probabilities = 1 / (1 + self.fitness)
        probabilities /= probabilities.sum()
        parent_indices = np.random.choice(self.population_size, size=2, p=probabilities)
        return self.positions[parent_indices[0]], self.positions[parent_indices[1]]

    def __call__(self, func):
        self.positions = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * np.random.rand(self.population_size, self.dim)
        for i in range(self.population_size):
            self.fitness[i] = func(self.positions[i])
            if self.fitness[i] < self.best_fitness:
                self.best_individual = self.positions[i].copy()
                self.best_fitness = self.fitness[i]
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.best_individual

        while self.evaluations < self.budget:
            new_population = []
            elite_size = int(self.elite_fraction * self.population_size)
            elite_indices = np.argsort(self.fitness)[:elite_size]
            new_population.extend(self.positions[elite_indices])
            
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
                mutated_child = self.quantum_bit_flip(child)
                new_population.append(mutated_child)
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break

            self.positions = np.array(new_population)[:self.population_size]
            self.positions = np.clip(self.positions, func.bounds.lb, func.bounds.ub)

            for i in range(self.population_size):
                self.fitness[i] = func(self.positions[i])
                if self.fitness[i] < self.best_fitness:
                    self.best_individual = self.positions[i].copy()
                    self.best_fitness = self.fitness[i]

        return self.best_individual