import numpy as np

class EnhancedQuantumSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.alpha = 0.9
        self.beta = 1.5
        self.quantum_prob = 0.1
        self.velocities = np.random.rand(self.initial_population_size, self.dim) * 0.1

    def __call__(self, func):
        lower_bound = func.bounds.lb
        upper_bound = func.bounds.ub
        position = np.random.uniform(lower_bound, upper_bound, (self.initial_population_size, self.dim))
        personal_best_position = position.copy()
        personal_best_value = np.full(self.initial_population_size, float('inf'))

        global_best_position = None
        global_best_value = float('inf')

        evaluations = 0

        while evaluations < self.budget:
            # Adjust population size dynamically
            adaptive_population_size = max(10, int(self.initial_population_size * (1 - evaluations / self.budget)))

            for i in range(adaptive_population_size):
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

            for i in range(adaptive_population_size):
                r1, r2 = np.random.rand(2)
                self.velocities[i] = (self.alpha * self.velocities[i] +
                                      self.beta * r1 * (personal_best_position[i] - position[i]) +
                                      self.beta * r2 * (global_best_position - position[i]))

                if np.random.rand() < (self.quantum_prob * (1 - evaluations / self.budget)):
                    position[i] = np.random.uniform(lower_bound, upper_bound, self.dim)
                else:
                    position[i] = position[i] + self.velocities[i] * (1 - evaluations / self.budget)

                position[i] = np.clip(position[i], lower_bound, upper_bound)

            if evaluations % (self.budget // 10) == 0:
                self.alpha *= 0.95
                self.beta *= 0.97

        return global_best_position, global_best_value