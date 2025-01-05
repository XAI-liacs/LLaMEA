import numpy as np

class AdaptiveSwarmQuantumAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.weight = 0.8
        self.crossover_rate = 0.9

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
                # Differential Evolution: Mutation and Crossover
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = position_population[indices]
                mutant = x1 + self.weight * (x2 - x3)
                mutant = np.clip(mutant, lb, ub)
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, position_population[i])
                
                # Evaluate new trial
                trial_fitness = func(trial)
                evaluations += 1

                # Selection: keep the better solution
                if trial_fitness < fitness[i]:
                    position_population[i] = trial
                    fitness[i] = trial_fitness

                # Update best position
                if trial_fitness < fitness[best_index]:
                    best_index = i
                    best_position = trial

                if evaluations >= self.budget:
                    break

            # Update quantum representation towards best
            for i in range(self.population_size):
                quantum_population[i] = self.update_quantum_bits(quantum_population[i], quantum_population[best_index])

        return best_position, fitness[best_index]

    def quantum_to_position(self, quantum_bits, lb, ub):
        # Convert quantum bits to classical positions in the search space
        position = lb + quantum_bits * (ub - lb)
        return position

    def update_quantum_bits(self, quantum_bits, best_quantum_bits):
        # Simple attraction mechanism towards the best solution
        attraction_strength = 0.1
        updated_bits = quantum_bits + attraction_strength * (best_quantum_bits - quantum_bits)
        updated_bits = np.clip(updated_bits, 0, 1)
        return updated_bits