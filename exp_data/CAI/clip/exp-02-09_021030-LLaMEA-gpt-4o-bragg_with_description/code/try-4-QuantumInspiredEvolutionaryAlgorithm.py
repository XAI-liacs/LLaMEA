import numpy as np

class QuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.q_population = np.random.rand(self.population_size, self.dim, 2)
        self.best_solution = None

    def initialize_q_population(self):
        self.q_population = np.random.rand(self.population_size, self.dim, 2)
        self.q_population /= np.linalg.norm(self.q_population, axis=2, keepdims=True)

    def q_measure(self):
        return np.array([np.random.choice([0, 1], size=self.dim, p=[1 - q[0]**2, q[0]**2]) for q in self.q_population])

    def update_q_population(self, classical_population, best_index):
        for i in range(self.population_size):
            for j in range(self.dim):
                theta = 0.01 * (2 * classical_population[best_index, j] - 1) * (2 * classical_population[i, j] - 1)
                self.q_population[i, j] = self.rotate_quantum_bit(self.q_population[i, j], theta)

    def rotate_quantum_bit(self, q_bit, theta):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        new_q_bit = np.array([cos_theta * q_bit[0] - sin_theta * q_bit[1],
                              sin_theta * q_bit[0] + cos_theta * q_bit[1]])
        return new_q_bit / np.linalg.norm(new_q_bit)

    def __call__(self, func):
        bounds = func.bounds
        evaluations = 0
        best_score = float('inf')
        self.initialize_q_population()

        while evaluations < self.budget:
            classical_population = self.q_measure()
            classical_population = bounds.lb + (bounds.ub - bounds.lb) * classical_population
            scores = np.array([func(individual) for individual in classical_population])
            evaluations += self.population_size

            best_index = np.argmin(scores)
            if scores[best_index] < best_score:
                best_score = scores[best_index]
                self.best_solution = classical_population[best_index]

            self.update_q_population(classical_population, best_index)

        return self.best_solution