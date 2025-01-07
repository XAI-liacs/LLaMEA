import numpy as np

class QuantumInspiredParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(50, budget // 10)
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.alpha = 0.5  # Quantum cloud influence
        self.beta = 0.5   # Memory influence

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_position = lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)
        personal_best_positions = np.copy(pop_position)
        personal_best_fitness = np.full(self.pop_size, float('inf'))

        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.pop_size):
                fitness = func(pop_position[i])
                evaluations += 1

                if fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best_positions[i] = pop_position[i]

                if fitness < self.best_global_fitness:
                    self.best_global_fitness = fitness
                    self.best_global_position = pop_position[i]

                if evaluations >= self.budget:
                    break

            # Quantum-inspired position update
            for i in range(self.pop_size):
                # Generate a probability cloud around the best global position
                q_cloud = self.alpha * np.random.normal(0, 1, self.dim)
                # Memory component encourages retention of old information
                memory_component = self.beta * (personal_best_positions[i] - pop_position[i])
                # Compute new position
                pop_position[i] = self.best_global_position + q_cloud + memory_component
                pop_position[i] = np.clip(pop_position[i], lb, ub)

        return self.best_global_position, self.best_global_fitness