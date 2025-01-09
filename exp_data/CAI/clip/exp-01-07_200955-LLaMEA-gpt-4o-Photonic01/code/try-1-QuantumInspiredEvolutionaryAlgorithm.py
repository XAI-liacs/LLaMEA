import numpy as np

class QuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(10 * dim, 100)
        self.alpha = 0.5  # Entanglement parameter
        self.beta = 0.05  # Mutation rate
        self.current_evaluations = 0

    def initialize_population(self, bounds):
        lower, upper = bounds.lb, bounds.ub
        return np.random.rand(self.population_size, self.dim) * (upper - lower) + lower

    def evaluate(self, func, population):
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += len(population)
        return fitness

    def quantum_superposition(self, population):
        q_population = np.cos(population) + 1j * np.sin(population)
        return q_population

    def collapse_wave_function(self, q_population, bounds):
        angles = np.angle(q_population)
        real_population = (angles / (2 * np.pi)) * (bounds.ub - bounds.lb) + bounds.lb
        return np.real(real_population)

    def entangle(self, q_population):
        num_pairs = self.population_size // 2
        indices = np.random.permutation(self.population_size)
        for i in range(num_pairs):
            idx1, idx2 = indices[2*i], indices[2*i + 1]
            phi = 2 * np.pi * np.random.rand()
            q_population[idx1] = self.alpha * q_population[idx1] + (1 - self.alpha) * np.exp(1j * phi) * q_population[idx2]
            q_population[idx2] = self.alpha * q_population[idx2] + (1 - self.alpha) * np.exp(-1j * phi) * q_population[idx1]
        return q_population

    def mutate(self, q_population):
        mutation_mask = np.random.rand(self.population_size, self.dim) < self.beta
        q_population[mutation_mask] *= np.exp(1j * np.pi * (2 * np.random.rand() - 1))
        return q_population

    def optimize(self, func, bounds):
        population = self.initialize_population(bounds)
        fitness = self.evaluate(func, population)
        
        q_population = self.quantum_superposition(population)

        while self.current_evaluations < self.budget:
            q_population = self.entangle(q_population)
            q_population = self.mutate(q_population)

            population = self.collapse_wave_function(q_population, bounds)
            new_fitness = self.evaluate(func, population)

            better = new_fitness < fitness
            fitness[better] = new_fitness[better]
            population[better] = population[better]

            if self.current_evaluations >= self.budget:
                break

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]

    def __call__(self, func):
        bounds = func.bounds
        best_solution, best_value = self.optimize(func, bounds)
        return best_solution, best_value