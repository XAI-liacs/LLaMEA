import numpy as np
from scipy.optimize import rosen

class Quantum_Chaotic_ES:
    def __init__(self, budget, dim, population_size=50, omega=0.5, alpha=1.5, beta=1.5, quantum_prob=0.3):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.quantum_prob = quantum_prob
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_global_position = None
        best_global_value = float('inf')

        population = self.initialize_population(lb, ub)
        velocities = self.initialize_velocities()

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if np.random.rand() < self.quantum_prob:
                    population[i] = self.quantum_perturbation(population[i], lb, ub)

                chaotic_factor = self.chaotic_map(i)
                velocities[i] = (self.omega * velocities[i] +
                                 self.alpha * chaotic_factor * (best_global_position - population[i]) +
                                 self.beta * np.random.random(self.dim) * (population[np.random.randint(self.population_size)] - population[i]))

                population[i] = np.clip(population[i] + velocities[i], lb, ub)

                value = func(population[i])
                self.evaluations += 1

                if value < best_global_value:
                    best_global_value = value
                    best_global_position = population[i]

                if self.evaluations >= self.budget:
                    break

        return best_global_position

    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def initialize_velocities(self):
        return np.random.uniform(-1, 1, (self.population_size, self.dim))

    def quantum_perturbation(self, position, lb, ub):
        q_position = position + (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.05
        return np.clip(q_position, lb, ub)

    def chaotic_map(self, index):
        x = np.sin(np.pi * index / self.population_size)
        return (2 * x * (1 - x)) * (np.sin(self.evaluations / self.budget * np.pi) ** 2)