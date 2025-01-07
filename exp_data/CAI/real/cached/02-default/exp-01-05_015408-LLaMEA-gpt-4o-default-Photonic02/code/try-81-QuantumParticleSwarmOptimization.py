import numpy as np

class QuantumParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 10 * dim
        self.alpha = 0.5  # Exploration factor
        self.phi_p = 1.5  # Personal influence
        self.phi_g = 1.5  # Global influence
        self.quantum_adjustment = 0.1
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        quantum_states = np.random.rand(self.swarm_size, self.dim)
        positions = self.quantum_to_position(quantum_states, lb, ub)
        velocities = np.random.rand(self.swarm_size, self.dim) * (ub - lb) / 10
        fitness = np.array([func(pos) for pos in positions])
        evaluations = self.swarm_size
        personal_best_positions = np.copy(positions)
        personal_best_fitness = np.copy(fitness)
        global_best_index = np.argmin(fitness)
        global_best_position = positions[global_best_index]

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                # Update velocities based on personal and global best
                inertia = velocities[i]
                cognitive_component = self.phi_p * np.random.rand(self.dim) * (personal_best_positions[i] - positions[i])
                social_component = self.phi_g * np.random.rand(self.dim) * (global_best_position - positions[i])
                velocities[i] = inertia + cognitive_component + social_component
                
                # Quantum state adjustment for balance
                quantum_adjustment = self.alpha * (global_best_position - positions[i]) * self.quantum_adjustment
                quantum_states[i] = np.clip(quantum_states[i] + quantum_adjustment, 0, 1)
                
                # Update positions
                positions[i] = self.quantum_to_position(quantum_states[i], lb, ub) + velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)

                # Evaluate new position
                new_fitness = func(positions[i])
                evaluations += 1

                # Update personal best
                if new_fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = new_fitness
                    personal_best_positions[i] = positions[i]

                # Update global best
                if new_fitness < fitness[global_best_index]:
                    global_best_index = i
                    global_best_position = positions[i]

                if evaluations >= self.budget:
                    break

        return global_best_position, fitness[global_best_index]

    def quantum_to_position(self, quantum_bits, lb, ub):
        position = lb + quantum_bits * (ub - lb)
        return position