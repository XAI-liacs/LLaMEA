import numpy as np

class QIGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7

    def initialize(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, float('inf'))
        self.best_solution = None
        self.best_score = float('inf')

    def quantum_crossover(self, parent1, parent2):
        phi = np.random.uniform(0, 1, self.dim)
        offspring = phi * parent1 + (1 - phi) * parent2
        return np.clip(offspring, bounds.lb, bounds.ub)

    def mutate(self, individual, bounds):
        mutation_vector = np.random.normal(0, 1, self.dim)
        mutate_condition = np.random.rand(self.dim) < self.mutation_rate
        individual[mutate_condition] += mutation_vector[mutate_condition]
        return np.clip(individual, bounds.lb, bounds.ub)

    def select_parents(self):
        selected_indices = np.random.choice(self.population_size, 2, replace=False)
        return self.population[selected_indices[0]], self.population[selected_indices[1]]

    def __call__(self, func):
        self.func = func
        bounds = func.bounds
        self.initialize(bounds)

        evaluations = 0
        while evaluations < self.budget:
            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = self.select_parents()

                if np.random.rand() < self.crossover_rate:
                    offspring1 = self.quantum_crossover(parent1, parent2)
                    offspring2 = self.quantum_crossover(parent2, parent1)
                else:
                    offspring1, offspring2 = parent1, parent2

                offspring1 = self.mutate(offspring1, bounds)
                offspring2 = self.mutate(offspring2, bounds)

                new_population.extend([offspring1, offspring2])

            self.population = np.array(new_population)

            for i in range(self.population_size):
                score = self.func(self.population[i])
                evaluations += 1
                if score < self.fitness[i]:
                    self.fitness[i] = score

                if score < self.best_score:
                    self.best_score = score
                    self.best_solution = self.population[i]

        return self.best_solution, self.best_score