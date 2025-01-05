import numpy as np

class QuantumInspiredSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.inertia_weight = 0.9
        self.cognitive_coeff = 2.0
        self.swarm_coeff = 2.0
        self.inertia_decay = 0.99

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        quantum_population = np.random.rand(self.population_size, self.dim)
        velocity = np.random.rand(self.population_size, self.dim) * 0.1
        position_population = self.quantum_to_position(quantum_population, lb, ub)
        personal_best_position = np.copy(position_population)
        fitness = np.array([func(ind) for ind in position_population])
        personal_best_fitness = np.copy(fitness)
        evaluations = self.population_size
        global_best_index = np.argmin(fitness)
        global_best_position = position_population[global_best_index]

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Update velocity considering inertia, cognitive, and swarm components
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.cognitive_coeff * r1 * (personal_best_position[i] - position_population[i])
                swarm_component = self.swarm_coeff * r2 * (global_best_position - position_population[i])
                
                velocity[i] = (self.inertia_weight * velocity[i] + cognitive_component + swarm_component)
                velocity[i] = np.clip(velocity[i], -1, 1)

                # Update position with velocity and quantum-inspired influence
                quantum_population[i] = self.update_quantum_bits(quantum_population[i], velocity[i])
                position_population[i] = self.quantum_to_position(quantum_population[i], lb, ub)

                # Evaluate new position
                new_fitness = func(position_population[i])
                evaluations += 1

                # Update personal best
                if new_fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = new_fitness
                    personal_best_position[i] = position_population[i]

                # Update global best
                if new_fitness < personal_best_fitness[global_best_index]:
                    global_best_index = i
                    global_best_position = position_population[i]

                if evaluations >= self.budget:
                    break

            # Decay inertia weight to balance exploration and exploitation
            self.inertia_weight *= self.inertia_decay

        return global_best_position, personal_best_fitness[global_best_index]

    def quantum_to_position(self, quantum_bits, lb, ub):
        return lb + quantum_bits * (ub - lb)

    def update_quantum_bits(self, quantum_bits, velocity):
        new_quantum_bits = quantum_bits + velocity
        return np.clip(new_quantum_bits, 0, 1)