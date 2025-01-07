import numpy as np

class QuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(50, budget // 10)
        self.q_population = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.alpha = 0.01  # Learning rate for updating quantum bits

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        evaluations = 0
        
        while evaluations < self.budget:
            positions = np.clip(self.quantum_to_real(self.q_population), lb, ub)
            fitness_values = np.array([func(pos) for pos in positions])
            evaluations += self.pop_size

            min_idx = np.argmin(fitness_values)
            
            if fitness_values[min_idx] < self.best_fitness:
                self.best_fitness = fitness_values[min_idx]
                self.best_solution = positions[min_idx]
            
            mean_fitness = np.mean(fitness_values)
            self.update_quantum_bits(fitness_values, mean_fitness)

        return self.best_solution, self.best_fitness

    def quantum_to_real(self, q_population):
        return (ub + lb) / 2 + (ub - lb) / 2 * np.tanh(q_population)

    def update_quantum_bits(self, fitness_values, mean_fitness):
        for i in range(self.pop_size):
            if fitness_values[i] < mean_fitness:
                rotation_angle = self.alpha
            else:
                rotation_angle = -self.alpha

            # Update quantum bits using a rotation gate concept
            self.q_population[i] += rotation_angle * np.sign(self.q_population[i])
            np.clip(self.q_population[i], -1, 1, out=self.q_population[i])