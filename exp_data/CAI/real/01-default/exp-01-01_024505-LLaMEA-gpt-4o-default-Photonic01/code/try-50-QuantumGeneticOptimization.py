import numpy as np

class QuantumGeneticOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, min(50, budget // 10))
        self.population = None
        self.fitness = np.full(self.population_size, float('inf'))
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.crossover_prob = 0.8
        self.mutation_prob = 0.2
        self.interference_prob = 0.1

    def initialize_population(self, lb, ub):
        self.population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)

    def evaluate_population(self, func):
        fitness = np.array([func(individual) for individual in self.population])
        for i, f in enumerate(fitness):
            if f < self.fitness[i]:
                self.fitness[i] = f
            if f < self.global_best_fitness:
                self.global_best_fitness = f
                self.global_best_position = self.population[i]
        return fitness

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_prob:
            point = np.random.randint(1, self.dim - 1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1, parent2

    def mutate(self, individual, lb, ub):
        if np.random.rand() < self.mutation_prob:
            mutation_vector = np.random.randn(self.dim) * 0.1
            individual += mutation_vector
            individual = np.clip(individual, lb, ub)
        return individual

    def apply_quantum_interference(self, lb, ub):
        for i in range(self.population_size):
            if np.random.rand() < self.interference_prob:
                interference_vector = lb + (ub - lb) * np.random.rand(self.dim)
                self.population[i] = np.mean([self.population[i], interference_vector], axis=0)
                self.population[i] = np.clip(self.population[i], lb, ub)

    def select_parents(self):
        idx = np.random.choice(self.population_size, size=2, replace=False, p=self.fitness / np.sum(self.fitness))
        return self.population[idx[0]], self.population[idx[1]]

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            self.evaluate_population(func)
            evaluations += self.population_size

            if evaluations >= self.budget:
                break

            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = self.select_parents()
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1, lb, ub)
                child2 = self.mutate(child2, lb, ub)
                new_population.extend([child1, child2])
            self.population = np.array(new_population)
            self.apply_quantum_interference(lb, ub)

        return self.global_best_position, self.global_best_fitness