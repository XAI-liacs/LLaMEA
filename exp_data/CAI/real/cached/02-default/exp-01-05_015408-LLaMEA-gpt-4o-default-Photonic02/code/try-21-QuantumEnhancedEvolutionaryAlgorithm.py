import numpy as np

class QuantumEnhancedEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.alpha = 0.5
        self.beta = 0.5
        self.learning_rate_decay = 0.99  # Adaptive learning rate decay
        self.diversity_threshold = 0.1   # Threshold for diversity preservation

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        quantum_population = np.random.rand(self.population_size, self.dim)
        position_population = self.quantum_to_position(quantum_population, lb, ub)
        fitness = np.array([func(ind) for ind in position_population])
        evaluations = self.population_size
        best_index = np.argmin(fitness)
        best_position = position_population[best_index]

        while evaluations < self.budget:
            new_quantum_population = np.empty_like(quantum_population)
            for i in range(self.population_size):
                # Adaptive Quantum Rotation Update
                learning_rate = self.beta * (self.learning_rate_decay ** (evaluations / self.population_size))
                if np.random.rand() < self.alpha:
                    new_quantum_population[i] = self.update_quantum_bits(quantum_population[i], quantum_population[best_index], learning_rate)
                else:
                    new_quantum_population[i] = quantum_population[i]

                # Convert quantum representation to classical position
                position_population[i] = self.quantum_to_position(new_quantum_population[i], lb, ub)

                # Evaluate new position
                new_fitness = func(position_population[i])
                evaluations += 1

                # Selection: keep the better solution
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                    quantum_population[i] = new_quantum_population[i]

                # Update best position
                if new_fitness < fitness[best_index]:
                    best_index = i
                    best_position = position_population[i]

                if evaluations >= self.budget:
                    break

            # Diversity Preservation
            if self.calculate_diversity(quantum_population) < self.diversity_threshold:
                quantum_population += np.random.normal(0, 0.1, size=quantum_population.shape)

        return best_position, fitness[best_index]

    def quantum_to_position(self, quantum_bits, lb, ub):
        # Convert quantum bits to classical positions in the search space
        position = lb + quantum_bits * (ub - lb)
        return position

    def update_quantum_bits(self, quantum_bits, best_quantum_bits, learning_rate):
        # Quantum rotation gate inspired update with adaptive learning rate
        delta_theta = learning_rate * (best_quantum_bits - quantum_bits)
        new_quantum_bits = quantum_bits + delta_theta
        new_quantum_bits = np.clip(new_quantum_bits, 0, 1)
        return new_quantum_bits

    def calculate_diversity(self, quantum_population):
        # Calculate diversity as the average pairwise Euclidean distance
        num_individuals = quantum_population.shape[0]
        diversity_sum = 0
        count = 0
        for i in range(num_individuals):
            for j in range(i + 1, num_individuals):
                diversity_sum += np.linalg.norm(quantum_population[i] - quantum_population[j])
                count += 1
        return diversity_sum / count if count > 0 else 0