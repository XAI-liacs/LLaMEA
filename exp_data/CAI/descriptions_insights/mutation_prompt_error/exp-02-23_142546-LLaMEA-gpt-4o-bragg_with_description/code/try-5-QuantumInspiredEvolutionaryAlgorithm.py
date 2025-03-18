import numpy as np

class QuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.alpha = 0.5  # Influence of parent solution
        self.beta = 0.5   # Influence of global best solution
        self.best_global_position = None
        self.best_global_value = float('-inf')

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        evaluations = 0

        def quantum_bit_to_real(alpha, beta, parent, global_best):
            q = np.random.choice([alpha, beta], size=self.dim, p=[0.5, 0.5])
            return np.clip(q * parent + (1 - q) * global_best, lb, ub)

        while evaluations < self.budget:
            for i in range(self.population_size):
                parent = population[i]
                child = quantum_bit_to_real(self.alpha, self.beta, parent, self.best_global_position if self.best_global_position is not None else parent)
                value = func(child)
                evaluations += 1

                if value > self.best_global_value:
                    self.best_global_value = value
                    self.best_global_position = child

            population = np.array([quantum_bit_to_real(self.alpha, self.beta, ind, self.best_global_position) for ind in population])

        return self.best_global_position