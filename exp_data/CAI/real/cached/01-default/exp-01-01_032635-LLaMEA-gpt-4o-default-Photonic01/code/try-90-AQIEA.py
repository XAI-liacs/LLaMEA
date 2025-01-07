import numpy as np

class AQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget)
        self.positions = None
        self.fitness = None
        self.best_solution = None
        self.best_value = np.inf
        self.mutation_rate = 0.1
        self.adapt_rate = 0.1
        self.elite_fraction = 0.2

    def initialize_population(self, lb, ub):
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.bounds = (lb, ub)

    def adaptive_mutation(self, position):
        mutation_strength = np.random.normal(0, 0.1, self.dim)
        new_position = position + self.mutation_rate * mutation_strength * (self.bounds[1] - self.bounds[0])
        return np.clip(new_position, self.bounds[0], self.bounds[1])

    def quantum_exploration(self, elite_position):
        beta = np.random.normal(0, 1, self.dim)
        delta = np.random.normal(0, 1, self.dim) * 0.05
        new_position = elite_position + beta * (self.best_solution - elite_position) + delta
        return np.clip(new_position, self.bounds[0], self.bounds[1])

    def select_elites(self):
        elite_count = int(self.elite_fraction * self.population_size)
        elite_indices = np.argsort(self.fitness)[:elite_count]
        return self.positions[elite_indices]

    def evolve(self, func):
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
                    self.best_solution = self.positions[i].copy()

            elites = self.select_elites()

            for i in range(self.population_size):
                if np.random.rand() < self.adapt_rate:
                    self.positions[i] = self.adaptive_mutation(self.positions[i])
                if np.random.rand() < self.mutation_rate:
                    elite_choice = elites[np.random.randint(len(elites))]
                    self.positions[i] = self.quantum_exploration(elite_choice)

        return self.best_solution, self.best_value

    def __call__(self, func):
        return self.evolve(func)