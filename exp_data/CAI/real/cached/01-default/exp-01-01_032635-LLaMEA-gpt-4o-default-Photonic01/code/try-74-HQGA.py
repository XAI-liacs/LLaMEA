import numpy as np

class HQGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, budget)
        self.population = None
        self.fitness = None
        self.global_best_position = None
        self.global_best_value = np.inf
        self.quantum_prob = 0.1
        self.crossover_prob = 0.8
        self.mutation_prob = 0.05

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)

    def evaluate_population(self, func):
        for i in range(self.population_size):
            value = func(self.population[i])
            if value < self.fitness[i]:
                self.fitness[i] = value

            if value < self.global_best_value:
                self.global_best_value = value
                self.global_best_position = self.population[i].copy()

    def quantum_search(self, individual, lb, ub):
        beta = np.random.normal(0, 1, self.dim)
        delta = np.random.normal(0, 1, self.dim) * 0.05
        new_position = individual + beta * (self.global_best_position - individual) + delta
        return np.clip(new_position, lb, ub)

    def crossover(self, parent1, parent2):
        mask = np.random.rand(self.dim) < 0.5
        child = np.where(mask, parent1, parent2)
        return child

    def mutate(self, individual, lb, ub):
        mutation_vector = np.random.uniform(-0.1, 0.1, self.dim)
        new_individual = individual + mutation_vector
        return np.clip(new_individual, lb, ub)

    def select_parents(self):
        idx1, idx2 = np.random.choice(self.population_size, 2, replace=False)
        return self.population[idx1], self.population[idx2]

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            self.evaluate_population(func)
            evaluations += self.population_size

            new_population = []

            while len(new_population) < self.population_size:
                if evaluations >= self.budget:
                    break

                parent1, parent2 = self.select_parents()

                if np.random.rand() < self.quantum_prob:
                    child1 = self.quantum_search(parent1, lb, ub)
                    child2 = self.quantum_search(parent2, lb, ub)
                else:
                    if np.random.rand() < self.crossover_prob:
                        child1 = self.crossover(parent1, parent2)
                        child2 = self.crossover(parent2, parent1)
                    else:
                        child1, child2 = parent1, parent2

                if np.random.rand() < self.mutation_prob:
                    child1 = self.mutate(child1, lb, ub)
                if np.random.rand() < self.mutation_prob:
                    child2 = self.mutate(child2, lb, ub)

                new_population.extend([child1, child2])

            self.population = np.array(new_population[:self.population_size])

        return self.global_best_position, self.global_best_value