import numpy as np

class ImprovedQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def apply_gate(self, individual, mutation_rate):
        return individual + mutation_rate * individual * np.random.choice([-1, 1], size=self.dim)

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            best_idx = np.argmin(fitness)
            best_individual = self.population[best_idx]
            mutation_rate = 1 / (1 + np.max(fitness) - np.min(fitness))
            for idx in range(self.budget):
                self.population[idx] = 0.5 * (self.population[idx] + self.apply_gate(best_individual, mutation_rate))
        return best_individual