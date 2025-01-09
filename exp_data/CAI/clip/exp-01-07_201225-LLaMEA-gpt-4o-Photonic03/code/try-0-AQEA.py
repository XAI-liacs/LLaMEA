import numpy as np

class AQEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.quantum_population = np.random.rand(self.population_size, dim)
        self.best_solution = None
        self.best_fitness = np.inf

    def __call__(self, func):
        bounds = np.vstack((func.bounds.lb, func.bounds.ub)).T
        evaluations = 0

        def decode_population(quantum_population):
            return bounds[:, 0] + ((bounds[:, 1] - bounds[:, 0]) * quantum_population)

        def evaluate_population(population):
            nonlocal evaluations
            fitness = np.array([func(ind) for ind in population])
            evaluations += len(population)
            return fitness

        while evaluations < self.budget:
            population = decode_population(self.quantum_population)
            fitness = evaluate_population(population)

            min_fitness_idx = np.argmin(fitness)
            if fitness[min_fitness_idx] < self.best_fitness:
                self.best_fitness = fitness[min_fitness_idx]
                self.best_solution = population[min_fitness_idx]

            for i in range(self.population_size):
                phi = np.random.rand(self.dim)
                R = np.random.rand(self.dim)
                self.quantum_population[i] = (R < phi) * self.quantum_population[min_fitness_idx] + (R >= phi) * self.quantum_population[i]

                mutation_rate = 1.0 / (self.dim + np.sqrt(np.sum(self.quantum_population[i]**2)))
                mutation = mutation_rate * np.random.randn(self.dim)
                self.quantum_population[i] = np.clip(self.quantum_population[i] + mutation, 0, 1)

        return self.best_solution, self.best_fitness