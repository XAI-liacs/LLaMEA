import numpy as np

class ImprovedQuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.alpha = 0.5
        self.beta = 0.5
        self.adaptive_factor = 0.1
        self.crossover_prob = 0.7
        self.mutation_factor = 0.8

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        quantum_population = np.random.rand(self.population_size, self.dim)
        position_population = self.quantum_to_position(quantum_population, lb, ub)
        fitness = np.array([func(ind) for ind in position_population])
        evaluations = self.population_size
        best_index = np.argmin(fitness)
        best_position = position_population[best_index]

        while evaluations < self.budget:
            new_quantum_population = quantum_population.copy()
            for i in range(self.population_size):
                # Adaptive quantum rotation gate: adjust based on success rate
                if np.random.rand() < self.alpha:
                    new_quantum_population[i] = self.update_quantum_bits(quantum_population[i], quantum_population[best_index], fitness, i)

                # Differential Evolution crossover
                if np.random.rand() < self.crossover_prob:
                    r1, r2, r3 = np.random.choice(self.population_size, 3, replace=False)
                    trial_quantum_bits = quantum_population[r1] + self.mutation_factor * (quantum_population[r2] - quantum_population[r3])
                    trial_quantum_bits = np.clip(trial_quantum_bits, 0, 1)
                    new_quantum_population[i] = np.where(np.random.rand(self.dim) < self.crossover_prob, trial_quantum_bits, new_quantum_population[i])

                # Convert to classical position
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

        return best_position, fitness[best_index]

    def quantum_to_position(self, quantum_bits, lb, ub):
        # Convert quantum bits to classical positions in the search space
        position = lb + quantum_bits * (ub - lb)
        return position

    def update_quantum_bits(self, quantum_bits, best_quantum_bits, fitness, index):
        # Adaptive quantum rotation inspired update
        delta_theta = self.beta * (best_quantum_bits - quantum_bits)
        improvement_ratio = (fitness[index] - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + 1e-9)
        adaptive_delta = delta_theta * (1 + self.adaptive_factor * (0.5 - improvement_ratio))
        new_quantum_bits = quantum_bits + adaptive_delta
        new_quantum_bits = np.clip(new_quantum_bits, 0, 1)
        return new_quantum_bits