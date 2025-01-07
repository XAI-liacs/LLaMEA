import numpy as np

class QuantumInspiredOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(50, budget // 10)  # Adaptive population size
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.alpha = 0.8  # Influence coefficient of the best solution

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initialize population in quantum states (superposition of lb and ub)
        q_population = np.random.uniform(0, 1, (self.pop_size, self.dim))
        # Classical positions derived from quantum states
        pop_position = lb + (ub - lb) * q_population
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

            # Update quantum states using best solutions
            for i in range(self.pop_size):
                # Quantum-inspired update mechanism
                q_population[i] = self.alpha * q_population[i] + (1 - self.alpha) * np.random.rand(self.dim)
                pop_position[i] = lb + (ub - lb) * q_population[i]

                # Include a random entanglement-like behavior
                if np.random.rand() < 0.1:
                    indices = np.random.choice(self.pop_size, 2, replace=False)
                    q_population[i] = (q_population[indices[0]] + q_population[indices[1]]) / 2

        return self.best_global_position, self.best_global_fitness