import numpy as np

class RefinedQuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.alpha = 0.5
        self.beta = 0.5
        self.adaptive_factor = 0.1
        self.elite_reeval_factor = 0.05  # Fraction of budget used for re-evaluating elite solutions

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        quantum_population = np.random.rand(self.population_size, self.dim)
        position_population = self.quantum_to_position(quantum_population, lb, ub)
        fitness = np.array([func(ind) for ind in position_population])
        evaluations = self.population_size
        best_index = np.argmin(fitness)
        best_position = position_population[best_index]
        elite_count = int(self.elite_reeval_factor * self.budget)

        while evaluations < self.budget - elite_count:
            for i in range(self.population_size):
                if np.random.rand() < self.alpha:
                    quantum_population[i] = self.update_quantum_bits(quantum_population[i], quantum_population[best_index], fitness, i)
                if np.random.rand() < self.alpha / 2:  # Introduce dual quantum update for diversity
                    quantum_population[i] = self.update_quantum_bits(quantum_population[i], np.random.choice(quantum_population), fitness, i)

                position_population[i] = self.quantum_to_position(quantum_population[i], lb, ub)
                new_fitness = func(position_population[i])
                evaluations += 1

                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness

                if new_fitness < fitness[best_index]:
                    best_index = i
                    best_position = position_population[i]

                if evaluations >= self.budget - elite_count:
                    break

        reeval_indices = np.argsort(fitness)[:elite_count]
        for idx in reeval_indices:
            new_fitness = func(position_population[idx])
            evaluations += 1
            if new_fitness < fitness[idx]:
                fitness[idx] = new_fitness
                if new_fitness < fitness[best_index]:
                    best_index = idx
                    best_position = position_population[idx]

            if evaluations >= self.budget:
                break

        return best_position, fitness[best_index]

    def quantum_to_position(self, quantum_bits, lb, ub):
        position = lb + quantum_bits * (ub - lb)
        return position

    def update_quantum_bits(self, quantum_bits, ref_quantum_bits, fitness, index):
        delta_theta = self.beta * (ref_quantum_bits - quantum_bits)
        improvement_ratio = (fitness[index] - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + 1e-9)
        adaptive_delta = delta_theta * (1 + self.adaptive_factor * (0.5 - improvement_ratio))
        new_quantum_bits = quantum_bits + adaptive_delta
        return np.clip(new_quantum_bits, 0, 1)