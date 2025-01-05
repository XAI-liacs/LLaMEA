import numpy as np

class QuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(50, budget // 10)
        self.alpha = 0.5  # Quantum rotation angle
        self.best_fitness = float('inf')
        self.best_position = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initialize the quantum bit population in the superposition state
        q_population = np.random.uniform(0, 1, (self.pop_size, self.dim, 2))
        population = self.qbit_to_real(q_population, lb, ub)
        fitness = np.full(self.pop_size, float('inf'))

        evaluations = 0

        while evaluations < self.budget:
            # Evaluate fitness and update bests
            for i in range(self.pop_size):
                fitness[i] = func(population[i])
                evaluations += 1

                if fitness[i] < self.best_fitness:
                    self.best_fitness = fitness[i]
                    self.best_position = population[i]

                if evaluations >= self.budget:
                    break

            # Update quantum particles
            for i in range(self.pop_size):
                for d in range(self.dim):
                    theta = self.alpha * (fitness[i] - self.best_fitness)
                    if np.random.rand() < 0.5:  # Quantum rotation operator
                        q_population[i, d, 0] = q_population[i, d, 0] * np.cos(theta) - q_population[i, d, 1] * np.sin(theta)
                        q_population[i, d, 1] = q_population[i, d, 0] * np.sin(theta) + q_population[i, d, 1] * np.cos(theta)
                    else:
                        q_population[i, d, 0] = q_population[i, d, 0] * np.cos(-theta) - q_population[i, d, 1] * np.sin(-theta)
                        q_population[i, d, 1] = q_population[i, d, 0] * np.sin(-theta) + q_population[i, d, 1] * np.cos(-theta)

            # Collapse quantum bits to generate new real-valued individuals
            population = self.qbit_to_real(q_population, lb, ub)

        return self.best_position, self.best_fitness

    def qbit_to_real(self, q_population, lb, ub):
        # Convert quantum bits to real values in the search space
        real_population = np.zeros((self.pop_size, self.dim))
        for i in range(self.pop_size):
            for d in range(self.dim):
                real_population[i, d] = lb[d] + (ub[d] - lb[d]) * (q_population[i, d, 0] ** 2 + q_population[i, d, 1] ** 2)
        return real_population