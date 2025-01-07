import numpy as np

class QIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.population_size = 20
        self.alpha = 0.9  # Quantum probability amplitude
        self.beta = 0.1  # Perturbation factor

    def initialize_population(self, lb, ub):
        return lb + (ub - lb) * np.random.rand(self.population_size, self.dim)

    def quantum_update(self, particle, lb, ub):
        random_vector = np.random.rand(self.dim)
        quantum_position = np.where(random_vector < self.alpha,
                                    particle + self.beta * np.random.normal(0, 1, self.dim),
                                    lb + (ub - lb) * np.random.rand(self.dim))
        return np.clip(quantum_position, lb, ub)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        population = self.initialize_population(lb, ub)

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                quantum_particle = self.quantum_update(population[i], lb, ub)
                value = func(quantum_particle)
                evaluations += 1

                if value < self.best_value:
                    self.best_value = value
                    self.best_solution = quantum_particle

                # Update the original particle position based on quantum mechanism
                population[i] = quantum_particle

        return self.best_solution, self.best_value