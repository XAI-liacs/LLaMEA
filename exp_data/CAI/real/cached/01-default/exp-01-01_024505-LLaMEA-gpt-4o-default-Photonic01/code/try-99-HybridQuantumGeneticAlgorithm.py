import numpy as np

class HybridQuantumGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(20, min(100, budget // 5))
        self.population = None
        self.fitness = None
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.mutation_rate = 0.05
        self.crossover_rate = 0.8
        self.adaptive_factor = 0.1
        self.quantum_delta = 0.01
        self.tournament_size = 5

    def initialize_population(self, lb, ub):
        self.population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.fitness = np.full(self.population_size, float('inf'))

    def evaluate_population(self, func):
        for i in range(self.population_size):
            f = func(self.population[i])
            if f < self.fitness[i]:
                self.fitness[i] = f
                if f < self.global_best_fitness:
                    self.global_best_fitness = f
                    self.global_best_position = np.copy(self.population[i])

    def tournament_selection(self):
        selected_indices = np.zeros(self.population_size, dtype=int)
        for i in range(self.population_size):
            competitors = np.random.choice(self.population_size, self.tournament_size, replace=False)
            selected_indices[i] = min(competitors, key=lambda idx: self.fitness[idx])
        return selected_indices

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, self.dim - 1)
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
            return child1, child2
        else:
            return np.copy(parent1), np.copy(parent2)

    def mutate(self, individual, lb, ub):
        for i in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                individual[i] += self.adaptive_factor * np.random.randn() * (ub[i] - lb[i])
                individual[i] = np.clip(individual[i], lb[i], ub[i])

    def quantum_mutation(self, individual, lb, ub):
        if np.random.rand() < self.quantum_delta:
            interference_vector = lb + (ub - lb) * np.random.rand(self.dim)
            individual[:] = np.mean([individual, interference_vector], axis=0)
            individual[:] = np.clip(individual, lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            self.evaluate_population(func)
            evaluations += self.population_size

            if evaluations >= self.budget:
                break

            selected_indices = self.tournament_selection()
            new_population = np.zeros_like(self.population)

            for i in range(0, self.population_size, 2):
                parent1 = self.population[selected_indices[i]]
                parent2 = self.population[selected_indices[i + 1]]
                child1, child2 = self.crossover(parent1, parent2)
                self.mutate(child1, lb, ub)
                self.mutate(child2, lb, ub)
                self.quantum_mutation(child1, lb, ub)
                self.quantum_mutation(child2, lb, ub)
                new_population[i] = child1
                new_population[i + 1] = child2

            self.population = new_population
            self.mutation_rate *= 0.99  # Decaying mutation rate to enhance exploitation

        return self.global_best_position, self.global_best_fitness