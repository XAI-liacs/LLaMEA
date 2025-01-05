import numpy as np

class EnhancedQuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.alpha = 0.5
        self.beta_initial = 0.5
        self.beta_final = 0.1
        self.crossover_rate = 0.7

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
                    quantum_population[i] = self.update_quantum_bits(quantum_population[i], quantum_population[best_index], evaluations)

                if np.random.rand() < self.crossover_rate:
                    partner = np.random.randint(self.population_size)
                    quantum_population[i] = self.crossover_quantum_bits(quantum_population[i], quantum_population[partner])

                position_population[i] = self.quantum_to_position(quantum_population[i], lb, ub)
                new_fitness = func(position_population[i])
                evaluations += 1

                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness

                if new_fitness < fitness[best_index]:
                    best_index = i
                    best_position = position_population[i]

                if evaluations >= self.budget:
                    break

        return best_position, fitness[best_index]

    def quantum_to_position(self, quantum_bits, lb, ub):
        position = lb + quantum_bits * (ub - lb)
        return position

    def update_quantum_bits(self, quantum_bits, best_quantum_bits, evaluations):
        beta = self.adaptive_beta(evaluations)
        delta_theta = beta * (best_quantum_bits - quantum_bits)
        new_quantum_bits = quantum_bits + delta_theta
        new_quantum_bits = np.clip(new_quantum_bits, 0, 1)
        return new_quantum_bits

    def adaptive_beta(self, evaluations):
        progress = evaluations / self.budget
        return self.beta_initial - (self.beta_initial - self.beta_final) * progress

    def crossover_quantum_bits(self, quantum_bits1, quantum_bits2):
        mask = np.random.rand(self.dim) < 0.5
        new_quantum_bits = np.where(mask, quantum_bits1, quantum_bits2)
        return new_quantum_bits