import numpy as np

class HybridSwarmQuantumAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        self.quantum_alpha = 0.5

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particle_position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        particle_velocity = np.zeros((self.population_size, self.dim))
        particle_best_position = np.copy(particle_position)
        fitness = np.array([func(ind) for ind in particle_position])
        evaluations = self.population_size

        best_index = np.argmin(fitness)
        global_best_position = particle_position[best_index]

        while evaluations < self.budget:
            r1, r2 = np.random.rand(), np.random.rand()

            for i in range(self.population_size):
                # Update particle velocity and position using PSO formulae
                particle_velocity[i] = (
                    particle_velocity[i] +
                    self.c1 * r1 * (particle_best_position[i] - particle_position[i]) +
                    self.c2 * r2 * (global_best_position - particle_position[i])
                )
                particle_position[i] += particle_velocity[i]
                particle_position[i] = np.clip(particle_position[i], lb, ub)

                # Quantum-inspired update
                quantum_bits = self.position_to_quantum(particle_position[i], lb, ub)
                quantum_bits = self.quantum_update(quantum_bits, global_best_position, lb, ub)
                particle_position[i] = self.quantum_to_position(quantum_bits, lb, ub)

                # Evaluate new position
                new_fitness = func(particle_position[i])
                evaluations += 1

                # Update personal best
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                    particle_best_position[i] = particle_position[i]

                # Update global best
                if new_fitness < fitness[best_index]:
                    best_index = i
                    global_best_position = particle_position[i]

                if evaluations >= self.budget:
                    break

        return global_best_position, fitness[best_index]

    def position_to_quantum(self, position, lb, ub):
        return (position - lb) / (ub - lb)

    def quantum_to_position(self, quantum_bits, lb, ub):
        return lb + quantum_bits * (ub - lb)

    def quantum_update(self, quantum_bits, global_best_position, lb, ub):
        global_best_quantum = self.position_to_quantum(global_best_position, lb, ub)
        delta = self.quantum_alpha * (global_best_quantum - quantum_bits)
        new_quantum_bits = quantum_bits + delta
        new_quantum_bits = np.clip(new_quantum_bits, 0, 1)
        return new_quantum_bits