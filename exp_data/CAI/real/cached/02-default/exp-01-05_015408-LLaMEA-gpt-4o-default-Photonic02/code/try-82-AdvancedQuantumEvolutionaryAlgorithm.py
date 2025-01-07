import numpy as np

class AdvancedQuantumEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.alpha = 0.5
        self.beta = 0.5
        self.adaptive_factor = 0.1
        self.elite_fraction = 0.1  # Fraction of the population considered elite

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        quantum_population = np.random.rand(self.population_size, self.dim)
        position_population = self.quantum_to_position(quantum_population, lb, ub)
        fitness = np.array([func(ind) for ind in position_population])
        evaluations = self.population_size
        elite_size = max(1, int(self.elite_fraction * self.population_size))
        
        while evaluations < self.budget:
            elite_indices = np.argsort(fitness)[:elite_size]
            for i in range(self.population_size):
                if np.random.rand() < self.alpha:
                    quantum_population[i] = self.update_quantum_bits(quantum_population[i], quantum_population[elite_indices], fitness, i)

                position_population[i] = self.quantum_to_position(quantum_population[i], lb, ub)
                new_fitness = func(position_population[i])
                evaluations += 1

                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness

                if evaluations >= self.budget:
                    break

        best_index = np.argmin(fitness)
        return position_population[best_index], fitness[best_index]

    def quantum_to_position(self, quantum_bits, lb, ub):
        return lb + quantum_bits * (ub - lb)

    def update_quantum_bits(self, quantum_bits, elite_quantum_bits, fitness, index):
        elite_best_bits = elite_quantum_bits[np.random.choice(elite_quantum_bits.shape[0])]
        delta_theta = self.beta * (elite_best_bits - quantum_bits)
        improvement_ratio = (fitness[index] - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + 1e-9)
        adaptive_delta = delta_theta * (1 + self.adaptive_factor * (0.5 - improvement_ratio))
        new_quantum_bits = quantum_bits + adaptive_delta
        new_quantum_bits = np.clip(new_quantum_bits, 0, 1)
        return new_quantum_bits