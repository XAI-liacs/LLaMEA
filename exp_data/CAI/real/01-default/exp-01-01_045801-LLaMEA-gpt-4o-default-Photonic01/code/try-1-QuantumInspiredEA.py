import numpy as np

class QuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.alpha = 0.9  # Rotation angle multiplier
        self.population = None
        self.best_solution = None
        self.best_fitness = float('inf')

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        for i in range(self.population_size):
            fitness = self.evaluate(self.population[i])
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = self.population[i].copy()

    def evaluate(self, solution):
        return self.func(solution)

    def quantum_bit_representation(self, solution):
        return np.greater(solution, (self.lb + self.ub) / 2).astype(float)

    def quantum_rotation_gate(self, q_bit, target):
        delta_theta = self.alpha * (target - q_bit)
        return (q_bit + delta_theta) % 1.0

    def evolve_population(self):
        q_population = np.array([self.quantum_bit_representation(ind) for ind in self.population])
        target = self.quantum_bit_representation(self.best_solution)

        for i in range(self.population_size):
            q_population[i] = self.quantum_rotation_gate(q_population[i], target)

        new_population = self.lb + (self.ub - self.lb) * q_population
        for i in range(self.population_size):
            fitness = self.evaluate(new_population[i])
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = new_population[i].copy()

        self.population = new_population

    def __call__(self, func):
        self.func = func
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub
        evaluations = 0

        # Initialize population
        self.initialize_population(self.lb, self.ub)

        while evaluations < self.budget:
            self.evolve_population()
            evaluations += self.population_size  # Assuming a full evaluation per generation

        return {'solution': self.best_solution, 'fitness': self.best_fitness}