import numpy as np

class EnhancedQuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.alpha = 0.5
        self.beta = 0.5
        self.adaptive_factor = 0.1
        self.de_mutation_factor = 0.8
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
                # Apply adaptive quantum rotation
                if np.random.rand() < self.alpha:
                    quantum_population[i] = self.update_quantum_bits(quantum_population[i], quantum_population[best_index], fitness, i)

                # Apply DE-inspired mutation and crossover
                target_idx = np.random.randint(0, self.population_size)
                r1, r2, r3 = np.random.choice([idx for idx in range(self.population_size) if idx != target_idx], 3, replace=False)
                mutant = quantum_population[r1] + self.de_mutation_factor * (quantum_population[r2] - quantum_population[r3])
                mutant = np.clip(mutant, 0, 1)
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, quantum_population[target_idx])
                position_trial = self.quantum_to_position(trial, lb, ub)

                # Evaluate the trial solution
                new_fitness = func(position_trial)
                evaluations += 1

                # Selection: replace if trial is better
                if new_fitness < fitness[target_idx]:
                    quantum_population[target_idx] = trial
                    fitness[target_idx] = new_fitness

                # Update best position
                if new_fitness < fitness[best_index]:
                    best_index = target_idx
                    best_position = position_trial

                if evaluations >= self.budget:
                    break

        return best_position, fitness[best_index]

    def quantum_to_position(self, quantum_bits, lb, ub):
        position = lb + quantum_bits * (ub - lb)
        return position

    def update_quantum_bits(self, quantum_bits, best_quantum_bits, fitness, index):
        delta_theta = self.beta * (best_quantum_bits - quantum_bits)
        improvement_ratio = (fitness[index] - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + 1e-9)
        adaptive_delta = delta_theta * (1 + self.adaptive_factor * (0.5 - improvement_ratio))
        new_quantum_bits = quantum_bits + adaptive_delta
        new_quantum_bits = np.clip(new_quantum_bits, 0, 1)
        return new_quantum_bits