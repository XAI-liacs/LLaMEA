import numpy as np

class QGDO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(20, budget)
        self.population = None
        self.pop_values = None
        self.best_solution = None
        self.best_value = np.inf
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.quantum_step_size = 0.05

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.pop_values = np.full(self.population_size, np.inf)

    def quantum_mutation(self, candidate, lb, ub):
        step = np.random.normal(0, self.quantum_step_size, self.dim)
        mutated = candidate + step * np.sign(self.best_solution - candidate)
        return np.clip(mutated, lb, ub)

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            crossover_point = np.random.randint(1, self.dim - 1)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            return child1, child2
        else:
            return parent1, parent2

    def select_parents(self):
        idx = np.random.choice(self.population_size, 2, replace=False)
        return self.population[idx[0]], self.population[idx[1]]

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = self.select_parents()
                child1, child2 = self.crossover(parent1, parent2)

                if np.random.rand() < self.mutation_rate:
                    child1 = self.quantum_mutation(child1, lb, ub)

                if np.random.rand() < self.mutation_rate:
                    child2 = self.quantum_mutation(child2, lb, ub)

                new_population.extend([child1, child2])

            for i, individual in enumerate(new_population):
                if evaluations >= self.budget:
                    break
                current_value = func(individual)
                evaluations += 1

                if current_value < self.pop_values[i]:
                    self.pop_values[i] = current_value
                    self.population[i] = individual.copy()

                if current_value < self.best_value:
                    self.best_value = current_value
                    self.best_solution = individual.copy()

            self.population = np.array(new_population)

        return self.best_solution, self.best_value