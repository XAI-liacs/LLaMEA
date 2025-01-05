import numpy as np

class EnhancedQuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.alpha = 0.5
        self.beta = 0.5
        self.elite_rate = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        quantum_population = np.random.rand(self.population_size, self.dim)
        position_population = self.quantum_to_position(quantum_population, lb, ub)
        fitness = np.array([func(ind) for ind in position_population])
        evaluations = self.population_size
        elite_size = max(1, int(self.elite_rate * self.population_size))
        elite_indices = np.argsort(fitness)[:elite_size]
        elites = position_population[elite_indices]

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Quantum rotation gate: adaptively update quantum bits
                if np.random.rand() < self.alpha:
                    quantum_population[i] = self.adaptive_quantum_rotation(quantum_population[i], elites)

                # Convert quantum representation to classical position
                position_population[i] = self.quantum_to_position(quantum_population[i], lb, ub)

                # Evaluate new position
                new_fitness = func(position_population[i])
                evaluations += 1

                # Selection: keep the better solution
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness

                # Elite preservation
                if new_fitness < fitness[elite_indices[-1]]:
                    elite_indices[-1] = i
                    elites = position_population[elite_indices]
                    elite_indices = np.argsort(fitness)[:elite_size]

                if evaluations >= self.budget:
                    break

        best_index = np.argmin(fitness)
        return position_population[best_index], fitness[best_index]

    def quantum_to_position(self, quantum_bits, lb, ub):
        # Convert quantum bits to classical positions in the search space
        position = lb + quantum_bits * (ub - lb)
        return position

    def adaptive_quantum_rotation(self, quantum_bits, elites):
        # Adaptive quantum rotation gate inspired update using elites
        elite_mean = np.mean(elites, axis=0)
        delta_theta = self.beta * (elite_mean - quantum_bits)
        new_quantum_bits = quantum_bits + delta_theta
        new_quantum_bits = np.clip(new_quantum_bits, 0, 1)
        return new_quantum_bits