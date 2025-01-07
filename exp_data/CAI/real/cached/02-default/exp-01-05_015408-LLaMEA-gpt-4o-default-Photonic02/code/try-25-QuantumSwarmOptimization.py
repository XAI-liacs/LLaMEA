import numpy as np

class QuantumSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 10 * dim
        self.alpha = 0.7  # Learning factor
        self.beta = 0.3   # Entanglement influence

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        quantum_positions = np.random.rand(self.swarm_size, self.dim)
        classical_positions = self.quantum_to_classical(quantum_positions, lb, ub)
        velocities = np.zeros((self.swarm_size, self.dim))
        personal_best_positions = np.copy(classical_positions)
        personal_best_fitness = np.array([func(ind) for ind in classical_positions])
        evaluations = self.swarm_size
        global_best_index = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_index]

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                # Update velocity with quantum-inspired dynamics
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.alpha * velocities[i] +
                                 self.beta * r1 * (personal_best_positions[i] - classical_positions[i]) +
                                 self.beta * r2 * (global_best_position - classical_positions[i]))

                # Update quantum positions using velocities
                quantum_positions[i] = self.update_quantum_positions(quantum_positions[i], velocities[i])

                # Convert quantum representation to classical position
                classical_positions[i] = self.quantum_to_classical(quantum_positions[i], lb, ub)

                # Evaluate new position
                new_fitness = func(classical_positions[i])
                evaluations += 1

                # Update personal best
                if new_fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = new_fitness
                    personal_best_positions[i] = classical_positions[i]

                # Update global best
                if new_fitness < personal_best_fitness[global_best_index]:
                    global_best_index = i
                    global_best_position = classical_positions[i]

                if evaluations >= self.budget:
                    break

        return global_best_position, personal_best_fitness[global_best_index]

    def quantum_to_classical(self, quantum_positions, lb, ub):
        # Translate quantum positions into classical search space
        return lb + quantum_positions * (ub - lb)

    def update_quantum_positions(self, quantum_positions, velocities):
        # Update quantum positions with a quantum-inspired velocity adjustment
        new_quantum_positions = quantum_positions + velocities
        new_quantum_positions = np.clip(new_quantum_positions, 0, 1)
        return new_quantum_positions