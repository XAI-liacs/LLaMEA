import numpy as np

class QuantumEnhancedDynamicAdaptationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.alpha = 0.5  # Initial rotation angle
        self.beta = 0.5  # Initial adaptation rate
        self.adaptation_rate = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        quantum_population = np.random.rand(self.population_size, self.dim)
        position_population = self.quantum_to_position(quantum_population, lb, ub)
        fitness = np.array([func(ind) for ind in position_population])
        evaluations = self.population_size
        best_index = np.argmin(fitness)
        best_position = position_population[best_index]

        while evaluations < self.budget:
            for i in range(self.population_size):
                if np.random.rand() < self.alpha:
                    quantum_population[i] = self.update_quantum_bits(quantum_population[i], quantum_population[best_index])

                position_population[i] = self.quantum_to_position(quantum_population[i], lb, ub)
                new_fitness = func(position_population[i])
                evaluations += 1

                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness

                if new_fitness < fitness[best_index]:
                    best_index = i
                    best_position = position_population[i]
                    self.adapt_parameters(new_fitness, fitness[best_index])

                if evaluations >= self.budget:
                    break

        return best_position, fitness[best_index]

    def quantum_to_position(self, quantum_bits, lb, ub):
        position = lb + quantum_bits * (ub - lb)
        return position

    def update_quantum_bits(self, quantum_bits, best_quantum_bits):
        delta_theta = self.beta * (best_quantum_bits - quantum_bits)
        new_quantum_bits = quantum_bits + delta_theta
        new_quantum_bits = np.clip(new_quantum_bits, 0, 1)
        return new_quantum_bits

    def adapt_parameters(self, new_fitness, best_fitness):
        if new_fitness < best_fitness:
            self.alpha = min(self.alpha + self.adaptation_rate, 1.0)
            self.beta = max(self.beta - self.adaptation_rate, 0.1)
        else:
            self.alpha = max(self.alpha - self.adaptation_rate, 0.1)
            self.beta = min(self.beta + self.adaptation_rate, 1.0)