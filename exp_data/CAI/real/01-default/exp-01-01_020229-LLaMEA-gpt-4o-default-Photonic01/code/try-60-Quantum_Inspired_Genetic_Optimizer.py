import numpy as np

class Quantum_Inspired_Genetic_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.q_bits = np.random.uniform(-np.pi/4, np.pi/4, (self.population_size, self.dim))
        self.beta = 0.5  # Learning rate for quantum rotation

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        def decode(q_bits):
            return lb + (ub - lb) * (np.sin(q_bits) ** 2)

        def fitness(individual):
            return func(decode(individual))

        fitness_values = np.array([fitness(q) for q in self.q_bits])
        evaluations = self.population_size

        while evaluations < self.budget:
            # Selection
            sorted_indices = np.argsort(fitness_values)
            self.q_bits = self.q_bits[sorted_indices]
            fitness_values = fitness_values[sorted_indices]

            new_q_bits = np.copy(self.q_bits)

            # Crossover
            for i in range(0, self.population_size, 2):
                if np.random.rand() < self.crossover_rate:
                    idx1, idx2 = i, i + 1
                    point = np.random.randint(1, self.dim)
                    new_q_bits[idx1, point:], new_q_bits[idx2, point:] = (
                        new_q_bits[idx2, point:].copy(),
                        new_q_bits[idx1, point:].copy(),
                    )

            # Mutation
            for i in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    mutation_idx = np.random.randint(self.dim)
                    new_q_bits[i, mutation_idx] += self.beta * np.random.normal()

            # Evaluation
            fitness_values = np.array([fitness(q) for q in new_q_bits])
            evaluations += self.population_size

            if evaluations >= self.budget:
                break

            # Update quantum bits based on selection
            for i in range(self.population_size):
                if fitness_values[i] < fitness(decode(self.q_bits[i])):
                    self.q_bits[i] = new_q_bits[i]

        best_index = np.argmin(fitness_values)
        best_q_bits = self.q_bits[best_index]
        best_position = decode(best_q_bits)
        best_value = fitness(best_q_bits)

        return best_position, best_value