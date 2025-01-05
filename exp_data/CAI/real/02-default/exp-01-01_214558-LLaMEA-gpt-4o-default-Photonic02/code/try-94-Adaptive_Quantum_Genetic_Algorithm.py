import numpy as np

class Adaptive_Quantum_Genetic_Algorithm:
    def __init__(self, budget, dim, population_size=20, crossover_prob=0.7, mutation_prob=0.1, quantum_prob=0.2):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.quantum_prob = quantum_prob
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.initialize_population(self.population_size, lb, ub)
        best_position = None
        best_value = float('inf')
        
        while self.evaluations < self.budget:
            fitness = self.evaluate_population(population, func)
            if self.evaluations >= self.budget:
                break

            if min(fitness) < best_value:
                best_value = min(fitness)
                best_position = population[np.argmin(fitness)]

            new_population = self.selection(population, fitness)
            self.crossover(new_population, lb, ub)
            self.mutation(new_population, lb, ub)
            self.quantum_perturbation(new_population, lb, ub)

            population = np.array(new_population)

        return best_position

    def initialize_population(self, size, lb, ub):
        return np.random.uniform(lb, ub, (size, self.dim))

    def evaluate_population(self, population, func):
        fitness = []
        for individual in population:
            value = func(individual)
            self.evaluations += 1
            fitness.append(value)
            if self.evaluations >= self.budget:
                break
        return fitness

    def selection(self, population, fitness):
        selected_indices = np.argsort(fitness)[:self.population_size // 2]
        return [population[i] for i in selected_indices]

    def crossover(self, population, lb, ub):
        for i in range(0, len(population), 2):
            if np.random.rand() < self.crossover_prob and i + 1 < len(population):
                parent1, parent2 = population[i], population[i + 1]
                point = np.random.randint(1, self.dim)
                child1 = np.concatenate((parent1[:point], parent2[point:]))
                child2 = np.concatenate((parent2[:point], parent1[point:]))
                population[i], population[i + 1] = np.clip(child1, lb, ub), np.clip(child2, lb, ub)

    def mutation(self, population, lb, ub):
        for i in range(len(population)):
            if np.random.rand() < self.mutation_prob:
                mutation_vector = np.random.uniform(-0.1, 0.1, self.dim)
                population[i] = np.clip(population[i] + mutation_vector, lb, ub)

    def quantum_perturbation(self, population, lb, ub):
        for i in range(len(population)):
            if np.random.rand() < self.quantum_prob:
                q_position = population[i] + (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.1
                population[i] = np.clip(q_position, lb, ub)