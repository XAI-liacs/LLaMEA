import numpy as np

class AdaptiveQuantumGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_base_rate = 0.05
        self.rotation_angle_base = np.pi / 6
        self.crossover_rate = 0.7
        self.adaptation_factor = 0.1
        self.position = None
        self.fitness = None
        self.best_solution = None
        self.best_fitness = float('inf')

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.position = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.fitness = np.full(self.population_size, float('inf'))

    def evaluate(self, func):
        scores = np.array([func(p) for p in self.position])
        for i in range(self.population_size):
            if scores[i] < self.fitness[i]:
                self.fitness[i] = scores[i]
            if scores[i] < self.best_fitness:
                self.best_fitness = scores[i]
                self.best_solution = self.position[i]
        return scores

    def adaptive_mutation(self, position, current_iteration, max_iterations):
        mutation_rate = self.mutation_base_rate * (1 + self.adaptation_factor * current_iteration / max_iterations)
        mutation = np.random.randn(*position.shape) * mutation_rate
        return position + mutation

    def quantum_rotation(self, position):
        theta = self.rotation_angle_base * np.random.rand(*position.shape)
        rotated_position = position * np.cos(theta) + np.random.rand(*position.shape) * np.sin(theta)
        return rotated_position

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            mask = np.random.rand(self.dim) < 0.5
            offspring = np.where(mask, parent1, parent2)
        else:
            offspring = parent1
        return offspring

    def evolve(self, max_iterations):
        for iteration in range(max_iterations):
            new_population = []
            for i in range(self.population_size):
                parent1_idx = np.random.randint(0, self.population_size)
                parent2_idx = np.random.randint(0, self.population_size)
                parent1 = self.position[parent1_idx]
                parent2 = self.position[parent2_idx]

                child = self.crossover(parent1, parent2)
                child = self.adaptive_mutation(child, iteration, max_iterations)
                child = self.quantum_rotation(child)

                new_population.append(child)

            self.position = np.array(new_population)

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        max_iterations = self.budget // self.population_size
        while func_calls < self.budget:
            scores = self.evaluate(func)
            func_calls += self.population_size
            self.evolve(max_iterations)

        return self.best_solution, self.best_fitness