import numpy as np

class QSEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget)
        self.positions = None
        self.fitness = None
        self.best_position = None
        self.best_value = np.inf
        self.mutation_rate = 0.1
        self.history = []

    def initialize_population(self, lb, ub):
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.bounds = (lb, ub)

    def quantum_update(self, position, best):
        phi = np.random.uniform(0, 2 * np.pi, self.dim)
        radius = np.random.uniform(0, 1, self.dim) * np.abs(best - position)
        new_position = position + radius * np.cos(phi)
        lb, ub = self.bounds
        return np.clip(new_position, lb, ub)

    def adaptive_mutation(self, position):
        if np.random.rand() < self.mutation_rate:
            mutation_vector = np.random.normal(0, 0.1, self.dim)
            position += mutation_vector
        lb, ub = self.bounds
        return np.clip(position, lb, ub)

    def evolution_step(self):
        sorted_indices = np.argsort(self.fitness)
        elite_count = self.population_size // 5
        for i in range(elite_count, self.population_size):
            parent_1, parent_2 = np.random.choice(elite_count, 2, replace=False)
            crossover_point = np.random.randint(1, self.dim)
            self.positions[i][:crossover_point] = self.positions[sorted_indices[parent_1]][:crossover_point]
            self.positions[i][crossover_point:] = self.positions[sorted_indices[parent_2]][crossover_point:]
            self.positions[i] = self.adaptive_mutation(self.positions[i])

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

                if current_value < self.fitness[i]:
                    self.fitness[i] = current_value

                if current_value < self.best_value:
                    self.best_value = current_value
                    self.best_position = self.positions[i].copy()

            self.history.append(self.best_value)
            self.evolution_step()

            for i in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    self.positions[i] = self.quantum_update(self.positions[i], self.best_position)

        return self.best_position, self.best_value