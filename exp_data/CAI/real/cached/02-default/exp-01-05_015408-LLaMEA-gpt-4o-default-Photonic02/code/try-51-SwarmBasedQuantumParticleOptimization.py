import numpy as np

class SwarmBasedQuantumParticleOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 8 * dim
        self.alpha = 0.4
        self.beta = 0.6
        self.inertia_weight = 0.9
        self.cognitive = 1.5
        self.social = 1.5

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        quantum_swarm = np.random.rand(self.swarm_size, self.dim)
        velocity_swarm = np.zeros((self.swarm_size, self.dim))
        position_swarm = self.quantum_to_position(quantum_swarm, lb, ub)
        fitness = np.array([func(ind) for ind in position_swarm])
        evaluations = self.swarm_size
        
        personal_best_position = np.copy(position_swarm)
        personal_best_fitness = np.copy(fitness)
        global_best_index = np.argmin(fitness)
        global_best_position = position_swarm[global_best_index]

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                # Update velocity using cognitive and social components
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.cognitive * r1 * (personal_best_position[i] - position_swarm[i])
                social_component = self.social * r2 * (global_best_position - position_swarm[i])
                velocity_swarm[i] = (self.inertia_weight * velocity_swarm[i] +
                                     cognitive_component + social_component)
                
                # Update quantum bits with velocity influence
                if np.random.rand() < self.alpha:
                    quantum_swarm[i] = self.update_quantum_bits(quantum_swarm[i], global_best_position, velocity_swarm[i])

                # Convert quantum representation to classical position
                position_swarm[i] = self.quantum_to_position(quantum_swarm[i], lb, ub)

                # Evaluate new position
                new_fitness = func(position_swarm[i])
                evaluations += 1

                # Update personal best
                if new_fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = new_fitness
                    personal_best_position[i] = position_swarm[i]

                # Update global best
                if new_fitness < fitness[global_best_index]:
                    global_best_index = i
                    global_best_position = position_swarm[i]

                if evaluations >= self.budget:
                    break

        return global_best_position, personal_best_fitness[global_best_index]

    def quantum_to_position(self, quantum_bits, lb, ub):
        # Convert quantum bits to classical positions in the search space
        position = lb + quantum_bits * (ub - lb)
        return position

    def update_quantum_bits(self, quantum_bits, best_position, velocity):
        # Update quantum bits influenced by classical velocity
        delta_theta = self.beta * (best_position - quantum_bits) + velocity
        new_quantum_bits = quantum_bits + delta_theta
        new_quantum_bits = np.clip(new_quantum_bits, 0, 1)
        return new_quantum_bits