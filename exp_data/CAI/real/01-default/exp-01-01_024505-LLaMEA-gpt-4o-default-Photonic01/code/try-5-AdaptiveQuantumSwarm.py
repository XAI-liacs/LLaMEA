import numpy as np

class AdaptiveQuantumSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(50, budget // 10)
        self.alpha = 0.5  # Quantum probability control
        self.beta = 0.1  # Local enhancement factor
        self.best_global_position = None
        self.best_global_fitness = float('inf')

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_position = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        pop_fitness = np.full(self.pop_size, float('inf'))
        
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.pop_size):
                fitness = func(pop_position[i])
                evaluations += 1

                if fitness < pop_fitness[i]:
                    pop_fitness[i] = fitness

                if fitness < self.best_global_fitness:
                    self.best_global_fitness = fitness
                    self.best_global_position = pop_position[i]

                if evaluations >= self.budget:
                    break

            # Quantum-inspired update
            for i in range(self.pop_size):
                if np.random.rand() < self.alpha:
                    # Generate superposition state
                    superposed_state = np.random.uniform(lb, ub, self.dim)
                    pop_position[i] = superposed_state
                else:
                    # Local enhancement using a Gaussian perturbation
                    perturbation = np.random.normal(0, self.beta, self.dim)
                    pop_position[i] += perturbation
                    np.clip(pop_position[i], lb, ub, out=pop_position[i])

        return self.best_global_position, self.best_global_fitness