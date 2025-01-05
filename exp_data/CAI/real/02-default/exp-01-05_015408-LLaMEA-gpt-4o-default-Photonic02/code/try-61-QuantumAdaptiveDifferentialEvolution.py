import numpy as np

class QuantumAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.alpha = 0.5
        self.beta = 0.5
        self.adaptive_factor = 0.1
        self.cr = 0.9  # Crossover probability
        self.f = 0.8   # Differential weight

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
                # Generate trial vector using Differential Evolution mechanism
                idxs = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = quantum_population[idxs[0]], quantum_population[idxs[1]], quantum_population[idxs[2]]
                mutant = self.mutate(a, b, c, self.f, best_index)
                trial = self.crossover(quantum_population[i], mutant, self.cr)
                
                # Convert quantum representation to classical position
                position_population[i] = self.quantum_to_position(trial, lb, ub)

                # Evaluate new position
                new_fitness = func(position_population[i])
                evaluations += 1

                # Selection: keep the better solution
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                    quantum_population[i] = trial

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

    def mutate(self, a, b, c, f, best_index):
        # Differential mutation with base vector selection
        mutant = a + f * (b - c)
        mutant = np.clip(mutant, 0, 1)
        return mutant

    def crossover(self, target, mutant, cr):
        # Uniform crossover
        crossover_mask = np.random.rand(self.dim) < cr
        trial = np.where(crossover_mask, mutant, target)
        return trial