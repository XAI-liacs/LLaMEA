import numpy as np

class EnhancedQuantumSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.alpha = 0.9
        self.beta = 1.5
        self.quantum_prob = 0.1  # Probability for quantum-inspired jump
        self.velocities = np.random.rand(self.population_size, self.dim) * 0.1
        self.mutation_rate = 0.05  # Added: Mutation rate for exploration

    def __call__(self, func):
        lower_bound = func.bounds.lb
        upper_bound = func.bounds.ub
        position = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        personal_best_position = position.copy()
        personal_best_value = np.full(self.population_size, float('inf'))

        global_best_position = None
        global_best_value = float('inf')

        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                current_value = func(position[i])
                evaluations += 1

                if current_value < personal_best_value[i]:
                    personal_best_value[i] = current_value
                    personal_best_position[i] = position[i]

                if current_value < global_best_value:
                    global_best_value = current_value
                    global_best_position = position[i]

                if evaluations >= self.budget:
                    break

            # Update velocities and positions
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                self.velocities[i] = (self.alpha * self.velocities[i] +
                                      self.beta * r1 * (personal_best_position[i] - position[i]) +
                                      self.beta * r2 * (global_best_position - position[i]))

                # Quantum-inspired jump with adaptive probability
                if np.random.rand() < (self.quantum_prob * (1 - evaluations / self.budget)):
                    position[i] = np.random.uniform(lower_bound, upper_bound, self.dim)
                else:
                    position[i] = position[i] + self.velocities[i] * (1 - evaluations / self.budget)

                # Mutation for diversified exploration
                if np.random.rand() < self.mutation_rate:
                    mutation_vector = np.random.normal(0, 0.1, self.dim)
                    position[i] += mutation_vector

                # Clamp positions to bounds
                position[i] = np.clip(position[i], lower_bound, upper_bound)

            # Adaptive parameter adjustment
            if evaluations % (self.budget // 10) == 0:
                self.alpha *= 0.95  # Decrease exploration over time
                self.beta *= 0.97  # Adjust convergence adaptively
                self.mutation_rate *= 0.95  # Reduce mutation over time

        return global_best_position, global_best_value