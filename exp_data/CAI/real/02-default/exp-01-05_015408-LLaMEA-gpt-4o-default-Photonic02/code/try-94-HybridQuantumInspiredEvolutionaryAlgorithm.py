import numpy as np

class HybridQuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.alpha = 0.5
        self.beta = 0.5
        self.adaptive_factor = 0.1
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.4
        self.social_coeff = 1.4
        self.velocity_clamp = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        quantum_population = np.random.rand(self.population_size, self.dim)
        position_population = self.quantum_to_position(quantum_population, lb, ub)
        velocity_population = np.random.uniform(-self.velocity_clamp, self.velocity_clamp, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in position_population])
        evaluations = self.population_size
        best_index = np.argmin(fitness)
        best_position = position_population[best_index]
        personal_best_positions = position_population.copy()
        personal_best_fitness = fitness.copy()

        while evaluations < self.budget:
            for i in range(self.population_size):
                # PSO velocity update
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_velocity = self.cognitive_coeff * r1 * (personal_best_positions[i] - position_population[i])
                social_velocity = self.social_coeff * r2 * (best_position - position_population[i])
                velocity_population[i] = (self.inertia_weight * velocity_population[i] +
                                          cognitive_velocity + social_velocity)
                velocity_population[i] = np.clip(velocity_population[i], -self.velocity_clamp, self.velocity_clamp)

                # PSO position update
                position_population[i] += velocity_population[i]
                position_population[i] = np.clip(position_population[i], lb, ub)

                # Adaptive quantum rotation gate
                if np.random.rand() < self.alpha:
                    quantum_population[i] = self.update_quantum_bits(quantum_population[i], quantum_population[best_index], fitness, i)

                # Convert quantum representation to classical position
                position_population[i] = self.quantum_to_position(quantum_population[i], lb, ub)

                # Evaluate new position
                new_fitness = func(position_population[i])
                evaluations += 1

                # Update personal best
                if new_fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = new_fitness
                    personal_best_positions[i] = position_population[i]

                # Update global best
                if new_fitness < fitness[best_index]:
                    best_index = i
                    best_position = position_population[i]

                if evaluations >= self.budget:
                    break

        return best_position, fitness[best_index]

    def quantum_to_position(self, quantum_bits, lb, ub):
        # Convert quantum bits to classical positions in the search space
        position = lb + quantum_bits * (ub - lb)
        return np.clip(position, lb, ub)

    def update_quantum_bits(self, quantum_bits, best_quantum_bits, fitness, index):
        # Adaptive quantum rotation inspired update
        delta_theta = self.beta * (best_quantum_bits - quantum_bits)
        improvement_ratio = (fitness[index] - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + 1e-9)
        adaptive_delta = delta_theta * (1 + self.adaptive_factor * (0.5 - improvement_ratio))
        new_quantum_bits = quantum_bits + adaptive_delta
        new_quantum_bits = np.clip(new_quantum_bits, 0, 1)
        return new_quantum_bits