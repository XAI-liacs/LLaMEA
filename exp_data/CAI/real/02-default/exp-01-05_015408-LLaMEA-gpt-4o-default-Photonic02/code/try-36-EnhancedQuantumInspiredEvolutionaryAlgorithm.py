import numpy as np

class EnhancedQuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.alpha = 0.5
        self.beta = 0.5
        self.diversity_threshold = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        quantum_population = np.random.rand(self.population_size, self.dim)
        position_population = self.quantum_to_position(quantum_population, lb, ub)
        fitness = np.array([func(ind) for ind in position_population])
        evaluations = self.population_size
        best_index = np.argmin(fitness)
        best_position = position_population[best_index]
        best_fitness = fitness[best_index]

        while evaluations < self.budget:
            for i in range(self.population_size):
                if np.random.rand() < self.alpha:
                    quantum_population[i] = self.adaptive_update_quantum_bits(quantum_population[i], quantum_population[best_index], fitness[i], best_fitness)

                position_population[i] = self.quantum_to_position(quantum_population[i], lb, ub)
                new_fitness = func(position_population[i])
                evaluations += 1

                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness

                if new_fitness < best_fitness:
                    best_index = i
                    best_position = position_population[i]
                    best_fitness = new_fitness

                if evaluations >= self.budget:
                    break

            if self.calculate_diversity(quantum_population) < self.diversity_threshold:
                quantum_population = self.randomize_population(quantum_population)

        return best_position, best_fitness

    def quantum_to_position(self, quantum_bits, lb, ub):
        position = lb + quantum_bits * (ub - lb)
        return position

    def adaptive_update_quantum_bits(self, quantum_bits, best_quantum_bits, current_fitness, best_fitness):
        adaptive_beta = self.beta * (current_fitness / best_fitness)
        delta_theta = adaptive_beta * (best_quantum_bits - quantum_bits)
        new_quantum_bits = quantum_bits + delta_theta
        new_quantum_bits = np.clip(new_quantum_bits, 0, 1)
        return new_quantum_bits

    def calculate_diversity(self, population):
        return np.mean(np.std(population, axis=0))

    def randomize_population(self, population):
        num_to_randomize = self.population_size // 2
        indices = np.random.choice(self.population_size, num_to_randomize, replace=False)
        for i in indices:
            population[i] = np.random.rand(self.dim)
        return population