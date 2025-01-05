import numpy as np

class HybridQuantumParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.alpha = 0.5  # Quantum rotation influence
        self.beta = 0.5   # Swarm velocity influence
        self.omega = 0.7  # Inertia weight for velocity
        self.phi_p = 1.5  # Personal best influence
        self.phi_g = 1.5  # Global best influence

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        quantum_population = np.random.rand(self.population_size, self.dim)
        position_population = self.quantum_to_position(quantum_population, lb, ub)
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        
        fitness = np.array([func(ind) for ind in position_population])
        evaluations = self.population_size
        
        personal_best_positions = np.copy(position_population)
        personal_best_fitness = np.copy(fitness)
        global_best_index = np.argmin(fitness)
        global_best_position = position_population[global_best_index]

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Update velocities based on personal and global bests
                r_p = np.random.rand(self.dim)
                r_g = np.random.rand(self.dim)
                velocity[i] = (
                    self.omega * velocity[i] +
                    self.phi_p * r_p * (personal_best_positions[i] - position_population[i]) +
                    self.phi_g * r_g * (global_best_position - position_population[i])
                )

                # Quantum rotation update
                if np.random.rand() < self.alpha:
                    quantum_population[i] = self.update_quantum_bits(quantum_population[i], quantum_population[global_best_index])

                # Calculate new positions
                position_population[i] = self.quantum_to_position(quantum_population[i], lb, ub) + self.beta * velocity[i]
                position_population[i] = np.clip(position_population[i], lb, ub)

                # Evaluate new position
                new_fitness = func(position_population[i])
                evaluations += 1
                
                # Update personal best
                if new_fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = new_fitness
                    personal_best_positions[i] = position_population[i]

                # Update global best
                if new_fitness < personal_best_fitness[global_best_index]:
                    global_best_index = i
                    global_best_position = position_population[i]

                if evaluations >= self.budget:
                    break

        return global_best_position, personal_best_fitness[global_best_index]

    def quantum_to_position(self, quantum_bits, lb, ub):
        position = lb + quantum_bits * (ub - lb)
        return position

    def update_quantum_bits(self, quantum_bits, best_quantum_bits):
        delta_theta = self.beta * (best_quantum_bits - quantum_bits)
        new_quantum_bits = quantum_bits + delta_theta
        new_quantum_bits = np.clip(new_quantum_bits, 0, 1)
        return new_quantum_bits