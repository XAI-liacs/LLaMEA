import numpy as np

class QuantumParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 10 * dim
        self.inertia_weight = 0.7
        self.cognitive_coefficient = 1.4
        self.social_coefficient = 1.4
        self.entanglement_factor = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        position = np.random.rand(self.swarm_size, self.dim) * (ub - lb) + lb
        velocity = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_position = np.copy(position)
        personal_best_fitness = np.array([func(ind) for ind in position])
        global_best_index = np.argmin(personal_best_fitness)
        global_best_position = personal_best_position[global_best_index]
        evaluations = self.swarm_size

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                # Update velocity with adaptive quantum entanglement
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_velocity = self.cognitive_coefficient * r1 * (personal_best_position[i] - position[i])
                social_velocity = self.social_coefficient * r2 * (global_best_position - position[i])
                velocity[i] = (self.inertia_weight * velocity[i] 
                               + cognitive_velocity 
                               + social_velocity 
                               + self.entanglement_factor * (np.random.rand(self.dim) - 0.5))

                # Update position
                position[i] = position[i] + velocity[i]
                position[i] = np.clip(position[i], lb, ub)

                # Evaluate new position
                fitness = func(position[i])
                evaluations += 1

                # Update personal best
                if fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best_position[i] = position[i]

                # Update global best
                if fitness < personal_best_fitness[global_best_index]:
                    global_best_index = i
                    global_best_position = position[i]

                if evaluations >= self.budget:
                    break

        return global_best_position, personal_best_fitness[global_best_index]