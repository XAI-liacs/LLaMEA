import numpy as np

class EnhancedQuantumInspiredOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(50, budget // 10)
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.alpha = 0.8

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        q_population = np.random.uniform(0, 1, (self.pop_size, self.dim))
        pop_position = lb + (ub - lb) * q_population
        personal_best_positions = np.copy(pop_position)
        personal_best_fitness = np.full(self.pop_size, float('inf'))

        evaluations = 0
        adapt_rate = 0.1  # Adaptive rate for entanglement

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

            for i in range(self.pop_size):
                q_population[i] = self.alpha * q_population[i] + (1 - self.alpha) * np.random.rand(self.dim)
                pop_position[i] = lb + (ub - lb) * q_population[i]

                if np.random.rand() < adapt_rate:
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                    q_population[i] = (q_population[indices[0]] + q_population[indices[1]] + q_population[indices[2]]) / 3

            adapt_rate = max(0.05, adapt_rate * 0.99)  # Gradually reduce adaptation rate

        return self.best_global_position, self.best_global_fitness