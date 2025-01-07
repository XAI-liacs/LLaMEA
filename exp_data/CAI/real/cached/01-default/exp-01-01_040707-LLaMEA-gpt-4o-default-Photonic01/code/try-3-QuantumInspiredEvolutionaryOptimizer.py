import numpy as np

class QuantumInspiredEvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.alpha = 0.1  # Quantum-inspired parameter
        self.mutation_rate = 0.1
        self.position = None
        self.best_position = None
        self.best_score = float('inf')

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.position = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.best_position = np.copy(self.position)

    def evaluate(self, func):
        scores = np.array([func(p) for p in self.position])
        for i in range(self.population_size):
            if scores[i] < self.best_score:
                self.best_score = scores[i]
                self.best_position = self.position[i]
        return scores

    def quantum_update(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        new_population = np.copy(self.position)
        for i in range(self.population_size):
            if np.random.rand() < self.alpha:
                quantum_factor = np.random.uniform(-0.5, 0.5, self.dim)
                new_population[i] = self.best_position + quantum_factor * (ub - lb)
            else:
                new_population[i] = self.position[i]
            new_population[i] = np.clip(new_population[i], lb, ub)
        return new_population

    def mutation(self, population, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        for i in range(self.population_size):
            if np.random.rand() < self.mutation_rate:
                mutation_vector = np.random.normal(0, 0.1, self.dim)
                population[i] = np.clip(population[i] + mutation_vector, lb, ub)
        return population

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        while func_calls < self.budget:
            scores = self.evaluate(func)
            func_calls += self.population_size
            quantum_population = self.quantum_update(func.bounds)
            self.position = self.mutation(quantum_population, func.bounds)
        return self.best_position, self.best_score