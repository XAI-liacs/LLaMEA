import numpy as np

class QuantumEntangledParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 10 * dim
        self.alpha = 0.5
        self.beta = 0.5
        self.entanglement_rate = 0.2

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        quantum_positions = np.random.rand(self.swarm_size, self.dim)
        velocities = np.random.rand(self.swarm_size, self.dim) * 0.1
        classical_positions = self.quantum_to_position(quantum_positions, lb, ub)
        fitness = np.array([func(ind) for ind in classical_positions])
        evaluations = self.swarm_size
        personal_best_positions = np.copy(classical_positions)
        personal_best_fitness = np.copy(fitness)
        global_best_index = np.argmin(fitness)
        global_best_position = classical_positions[global_best_index]

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                # Update velocities based on quantum entanglement and classical influences
                cognitive_component = self.alpha * np.random.rand(self.dim) * (personal_best_positions[i] - classical_positions[i])
                social_component = self.beta * np.random.rand(self.dim) * (global_best_position - classical_positions[i])
                quantum_entanglement = self.entanglement_rate * np.random.normal(0, 1, self.dim)

                velocities[i] = velocities[i] + cognitive_component + social_component + quantum_entanglement
                classical_positions[i] += velocities[i]
                classical_positions[i] = np.clip(classical_positions[i], lb, ub)

                # Evaluate new position
                new_fitness = func(classical_positions[i])
                evaluations += 1

                # Update personal and global bests
                if new_fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = new_fitness
                    personal_best_positions[i] = classical_positions[i]
                if new_fitness < fitness[global_best_index]:
                    global_best_index = i
                    global_best_position = classical_positions[i]

                if evaluations >= self.budget:
                    break

        return global_best_position, fitness[global_best_index]

    def quantum_to_position(self, quantum_bits, lb, ub):
        # Convert quantum bits to classical positions in the search space
        position = lb + quantum_bits * (ub - lb)
        return position