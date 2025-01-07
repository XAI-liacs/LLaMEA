import numpy as np

class AQDES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget)
        self.positions = None
        self.fitness_values = None
        self.best_position = None
        self.best_value = np.inf
        self.mutation_rate = 0.05
        self.bounds = None

    def initialize_population(self, lb, ub):
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.fitness_values = np.full(self.population_size, np.inf)
        self.bounds = (lb, ub)

    def quantum_position_update(self, individual):
        beta = np.random.normal(0, 1, self.dim)
        delta = np.random.normal(0, 1, self.dim)
        new_position = individual + beta * (self.best_position - individual) + delta * 0.1
        lb, ub = self.bounds
        return np.clip(new_position, lb, ub)

    def adaptive_mutation(self, individual):
        mutation_vector = np.random.uniform(-1, 1, self.dim) * self.mutation_rate
        return np.clip(individual + mutation_vector, self.bounds[0], self.bounds[1])

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                current_value = func(self.positions[i])
                evaluations += 1

                if current_value < self.fitness_values[i]:
                    self.fitness_values[i] = current_value

                if current_value < self.best_value:
                    self.best_value = current_value
                    self.best_position = self.positions[i].copy()

            for i in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    self.positions[i] = self.adaptive_mutation(self.positions[i])

                # Quantum-inspired position update
                if np.random.rand() < self.mutation_rate:
                    self.positions[i] = self.quantum_position_update(self.positions[i])

        return self.best_position, self.best_value