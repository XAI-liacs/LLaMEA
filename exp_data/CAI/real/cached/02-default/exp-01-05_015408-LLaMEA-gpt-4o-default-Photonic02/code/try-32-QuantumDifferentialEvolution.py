import numpy as np

class QuantumDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability

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
                # Mutation: create a donor vector using the best solution and two random vectors
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = indices
                if a == best_index:
                    a = (best_index + 1) % self.population_size
                donor_quantum_bits = self.quantum_mutation(quantum_population[a], quantum_population[b], quantum_population[c])

                # Crossover: mix donor vector with target vector
                trial_quantum_bits = self.quantum_crossover(quantum_population[i], donor_quantum_bits)

                # Convert quantum representation to classical position
                trial_position = self.quantum_to_position(trial_quantum_bits, lb, ub)

                # Evaluate new position
                new_fitness = func(trial_position)
                evaluations += 1

                # Selection: keep the better solution
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                    quantum_population[i] = trial_quantum_bits

                # Update best position
                if new_fitness < fitness[best_index]:
                    best_index = i
                    best_position = trial_position

                if evaluations >= self.budget:
                    break

        return best_position, fitness[best_index]

    def quantum_to_position(self, quantum_bits, lb, ub):
        # Convert quantum bits to classical positions in the search space
        position = lb + quantum_bits * (ub - lb)
        return position

    def quantum_mutation(self, a, b, c):
        # Differential mutation in quantum space
        mutant_quantum_bits = a + self.F * (b - c)
        mutant_quantum_bits = np.clip(mutant_quantum_bits, 0, 1)
        return mutant_quantum_bits

    def quantum_crossover(self, target, donor):
        # Crossover operation
        crossover_mask = np.random.rand(self.dim) < self.CR
        trial_quantum_bits = np.where(crossover_mask, donor, target)
        return trial_quantum_bits