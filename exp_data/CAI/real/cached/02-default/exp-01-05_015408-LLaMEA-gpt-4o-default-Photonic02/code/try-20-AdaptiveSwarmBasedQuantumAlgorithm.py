import numpy as np

class AdaptiveSwarmBasedQuantumAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.alpha = 0.5
        self.beta = 0.02
        self.inertia_weight = 0.9
        self.cognitive_coeff = 2.0
        self.social_coeff = 2.0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        quantum_population = np.random.rand(self.population_size, self.dim)
        position_population = self.quantum_to_position(quantum_population, lb, ub)
        velocity_population = np.random.rand(self.population_size, self.dim)
        fitness = np.array([func(ind) for ind in position_population])
        evaluations = self.population_size
        best_index = np.argmin(fitness)
        best_position = position_population[best_index]
        personal_best_positions = position_population.copy()
        personal_best_fitness = fitness.copy()

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Update velocity using PSO-like rules
                velocity_population[i] = (
                    self.inertia_weight * velocity_population[i]
                    + self.cognitive_coeff * np.random.rand(self.dim) * (personal_best_positions[i] - position_population[i])
                    + self.social_coeff * np.random.rand(self.dim) * (best_position - position_population[i])
                )
                velocity_population[i] = np.clip(velocity_population[i], -1, 1)

                # Update position from velocity
                position_population[i] += velocity_population[i]
                position_population[i] = np.clip(position_population[i], lb, ub)

                # Quantum-inspired update
                if np.random.rand() < self.alpha:
                    quantum_population[i] = self.update_quantum_bits(quantum_population[i], quantum_population[best_index])

                # Convert back to position and evaluate
                position_population[i] = self.quantum_to_position(quantum_population[i], lb, ub)
                new_fitness = func(position_population[i])
                evaluations += 1

                # Selection: update personal and global bests
                if new_fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = new_fitness
                    personal_best_positions[i] = position_population[i]

                if new_fitness < fitness[best_index]:
                    best_index = i
                    best_position = position_population[i]

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